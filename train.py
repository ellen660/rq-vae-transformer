import os
import argparse
import math

import rqvae.utils.dist as dist_utils
from rqvae.models import create_model
from rqvae.models.rqtransformer.configs import RQTransformerConfig, AttentionStackConfig, AttentionBlockConfig
from rqvae.my_code.shhs2_codes import Shhs2Dataset
from rqvae.my_code.metrics import Metrics, MetricsArgs
from rqvae.my_code.all_codes import AllCodes
from rqvae.my_code import MergedDataset
from rqvae.my_code.loss import compute_loss
from rqvae.optimizer import create_optimizer, create_scheduler
from rqvae.utils.utils import set_seed, compute_model_size, get_num_conv_linear_layers
from rqvae.my_code.schedulers import LinearWarmupCosineAnnealingLR
# from rqvae.utils.setup import setup

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import yaml
import random
from collections import defaultdict
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import numpy as np
from torch.cuda.amp import GradScaler, autocast
import time
import torch.nn.functional as F
import sys

from dataclasses import fields, is_dataclass
import torch.utils.checkpoint as checkpoint
import functools

# def checkpointed_forward(module, *inputs):
#     return checkpoint.checkpoint(module, *inputs)

# def checkpointed_forward(module, xs, cond=None, model_aux=None, amp=False, return_embeddings=False, one_hot=False):
    # """ Wrapper to handle optional non-tensor arguments. """
    # return module(xs, cond=cond, model_aux=model_aux, amp=amp, return_embeddings=return_embeddings, one_hot=one_hot)


#STEPS 
#Masked language modeling
    #Add a MASK token
    #Mask all 32 tokens 
    #Loss is different 
    #Remove mask for spatial 

def predict_future(model, logits, tau=0.1):
    #testing
    soft_tokens = F.gumbel_softmax(logits[:, 1:, :, :], tau, hard=False)
    hard_tokens = torch.argmax(logits[:, 1:, :, :], dim=-1)
    ste_tokens = (F.one_hot(hard_tokens.to(torch.int64), config.arch.vocab_size).float() - soft_tokens).detach() + soft_tokens
    logits2 = model(xs=ste_tokens, amp=config.common.amp, one_hot=True)
    return logits2

def train_one_step(metrics, epoch, optimizer, scheduler, model, train_loader, config, writer, scaler):
    model.train()
    epoch_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch}", unit="batch")

    for i, item in enumerate(progress_bar):
        x, y = item["x"], item["y"]
        x = x.to(device)
        if config.loss.soft:
            target = y.to(device)
        else:
            target = x
        # print(f'x shape: {x.shape} {x.dtype}') #B, T, D
        loss = 0
        
        #First pass
        logits = model(xs=x, amp=config.common.amp)  #B, T, D, vocab_size
        # print(f'logits {logits.shape}') #B, T, D, vocab_size
        loss += model.module.compute_loss(logits[:-1], target[1:], use_soft_target=config.loss.soft)

        #Predict future steps

        # for i in range(config.arch.num_steps-1):
            # logits_i = predict_future(model, logits, tau=0.1)
            # loss += model.module.compute_loss(logits_i, target[:, i+1:, :], use_soft_target=config.loss.soft)
            # logits = logits_i
            
        # Predict two steps ahead
        logits2 = predict_future(model, logits, tau=0.1)
        loss += model.module.compute_loss(logits2[:-1], target[:, 2:, :], use_soft_target=config.loss.soft)

        # Predict three steps ahead
        logits3 = predict_future(model, logits2, tau=0.1)
        loss += model.module.compute_loss(logits3[:-1], target[:, 3:, :], use_soft_target=config.loss.soft)

        # Predict four steps ahead
        logits4 = predict_future(model, logits3, tau=0.1)
        loss += model.module.compute_loss(logits4[:-1], target[:, 4:, :], use_soft_target=config.loss.soft)
            
        optimizer.zero_grad()  # Reset gradients
        # scaler.scale(loss).backward()  # Backpropagatio
        # scaler.unscale_(optimizer)
        loss.backward()

        # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)  # Prevent exploding gradients

        # scaler.step(optimizer)  # Update weights
        # scaler.update()  # Update scaler
        optimizer.step()

        epoch_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())
        losses = compute_loss(logits, x, soft=False)
        metrics.fill_metrics(losses, epoch*len(train_loader) + i)

    scheduler.step()  # Update learning rate
    loss_per_epoch = epoch_loss/len(train_loader)
    print(f"Epoch {epoch}, training loss: {loss_per_epoch}")
    metrics_dict = metrics.compute_and_log_metrics(loss_per_epoch)
    # log the learning rate
    metrics_dict['Learning Rate'] = optimizer.param_groups[0]['lr']
    logger(writer, metrics_dict, 'train', epoch)
    metrics.clear_metrics()

@torch.no_grad()
def test(metrics, epoch, model, val_loader, config, writer, scaler):
    model.eval()
    epoch_loss = 0

    progress_bar = tqdm(val_loader, desc=f"Validation Epoch {epoch}", unit="batch")

    for i, item in enumerate(progress_bar):
        x, y= item["x"], item["y"]
        x = x.to(device)
        if config.loss.soft:
            target = y.to(device)
        else:
            target = x

        logits = model(x)  # Forward pass
        loss = model.module.compute_loss(logits, target, use_soft_target=config.loss.soft)
        
        epoch_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())
        losses = compute_loss(logits, x, soft=False)
        metrics.fill_metrics(losses, epoch*len(val_loader) + i)

    loss_per_epoch = epoch_loss/len(val_loader)
    print(f"Epoch {epoch}, validation loss: {loss_per_epoch}")
    metrics_dict = metrics.compute_and_log_metrics(loss_per_epoch)
    # log the learning rate
    metrics_dict['Learning Rate'] = optimizer.param_groups[0]['lr']
    logger(writer, metrics_dict, 'val', epoch)
    metrics.clear_metrics()

def save_checkpoint(model, optimizer, scheduler, epoch, path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved at epoch {epoch}") 

def load_checkpoint(model, optimizer, scheduler, path, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch'] + 1  # Resume from next epoch
    # loss = checkpoint['loss']
    print(f"Checkpoint loaded: Resuming from epoch {epoch}")
    return epoch

#Logger for tensorboard
def logger(writer, metrics, phase, epoch_index):

    for key, value in metrics.items():

        if type(value)!= float and len(value.shape) > 0 and value.shape[0] == 2:
            value = value[1]
        elif type(value)!= float and len(value.shape) > 0 and value.shape[0] > 2:
            raise Exception("Need to handle multiclass")
            # bp()
        writer.add_scalar("%s/%s"%(phase, key), value, epoch_index)
    writer.flush()

class ConfigNamespace:
    """Converts a dictionary into an object-like namespace for easy attribute access."""
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                value = ConfigNamespace(value)  # Recursively convert nested dictionaries
            setattr(self, key, value)

def load_config(filepath, log_dir=None):
    #make directory
    with open(filepath, "r") as file:
        config_dict = yaml.safe_load(file)
        if log_dir:
            #save yaml file to log_dir
            with open(f"{log_dir}/config.yaml", "w") as file:
                yaml.dump(config_dict, file)
    return ConfigNamespace(config_dict)

def init_logger(log_dir, resume=False):
    print(f'log_dir: {log_dir}')
    
    if resume:
        # Resume logging
        writer = SummaryWriter(log_dir=log_dir, purge_step=None)  # Prevents overwriting
    else:
        writer = SummaryWriter(log_dir=log_dir)

    return writer

def init_dataset(config):
    train_dataset = Shhs2Dataset(mode="train", cv=config.dataset.cv, max_length=config.dataset.max_length, masking=config.dataset.masking)
    val_dataset = Shhs2Dataset(mode="val", cv=config.dataset.cv, max_length=config.dataset.max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=config.dataset.batch_size, shuffle=True, num_workers=config.dataset.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config.dataset.batch_size, shuffle=False, num_workers=config.dataset.num_workers)
    
    return train_loader, val_loader

# def init_dataset(config):
#     cv = config.dataset.cv
#     max_length = config.dataset.max_length
#     weights = {
#         "mgh": config.dataset.mgh,
#         "shhs2": config.dataset.shhs2,
#         "shhs1": config.dataset.shhs1,
#         "mros1": config.dataset.mros1,
#         "mros2": config.dataset.mros2,
#         "wsc": config.dataset.wsc,
#         "cfs": config.dataset.cfs,
#         "bwh": config.dataset.bwh,
#         "mesa": config.dataset.mesa
#     }

#     train_datasets = []
#     val_datasets = []
#     weight_list = []
#     # selected channels
#     channels = {}
#     if config.dataset.thorax > 0:
#         channels['thorax'] = config.dataset.thorax
#     if config.dataset.abdominal > 0:
#         channels['abdominal'] = config.dataset.abdominal

#     for dataset_name, weight in weights.items():
#         if weight > 0:
#             train_datasets.append(AllCodes(dataset=dataset_name, mode="train", cv=cv, channels=channels, max_length=max_length))
#             val_datasets.append(AllCodes(dataset=dataset_name, mode="val", cv=cv, channels=channels, max_length=max_length))
#             weight_list.append(weight)

#     print("Number of training datasets: ", len(train_datasets))
#     # merge the datasets
#     train_dataset = MergedDataset(train_datasets, weight_list, 1, debug = config.dataset.debug)
#     val_dataset = MergedDataset(val_datasets, weight_list, 0.2, config.dataset.debug)
#     train_loader = DataLoader(train_dataset, batch_size=config.dataset.batch_size, shuffle=True, num_workers=config.dataset.num_workers)
#     val_loader = DataLoader(val_dataset, batch_size=config.dataset.batch_size, shuffle=False, num_workers=config.dataset.num_workers)
#     print(f'Merged dataset size: {len(train_dataset)}')
#     print(f'weight {weights}')
#     return train_loader, val_loader

def init_model(config):
    def recursive_namespace_to_dataclass(namespace, dataclass_type):
        """Recursively converts a ConfigNamespace or dictionary into a dataclass instance."""
        
        if not is_dataclass(dataclass_type):
            raise TypeError(f"{dataclass_type} must be a dataclass")

        # Convert namespace to dictionary
        namespace_dict = vars(namespace) if not isinstance(namespace, dict) else namespace

        # Extract valid fields from the target dataclass
        dataclass_fields = {field.name: field.type for field in fields(dataclass_type)}

        # Construct dictionary with recursive handling for nested dataclasses
        filtered_dict = {}
        for key, field_type in dataclass_fields.items():
            if key in namespace_dict:
                value = namespace_dict[key]

                # If the field is another dataclass, apply recursion
                if is_dataclass(field_type):
                    filtered_dict[key] = recursive_namespace_to_dataclass(value, field_type)
                else:
                    filtered_dict[key] = value

        # Create an instance of the target dataclass
        return dataclass_type(**filtered_dict)

    rqtransformer = recursive_namespace_to_dataclass(config.arch, RQTransformerConfig)
    # breakpoint()
    model, model_ema = create_model(rqtransformer, ema=config.arch.ema is not None)
    # breakpoint()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model Total number of parameters: {total_params}")
    print(f'model {model}')
    # breakpoint()
    return model, model_ema

def set_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--config", type=str, default="test")
    parser.add_argument("--resume_from", type=str, default=f"/data/scratch/ellen660/rq-vae-transformer/tensorboard/test/20250217/101518/no")

    return parser.parse_args()

if __name__ == "__main__":

    args = set_args()
    user_name = os.getlogin()

    checkpoint_path = args.resume_from
    if os.path.exists(checkpoint_path):
        log_dir = checkpoint_path
        resume=True
        config = load_config(f"{checkpoint_path}/config.yaml")

    else:
        resume=False 
        config = load_config("rqvae/my_code/%s.yaml" % args.config)
        curr_time = datetime.now().strftime("%Y%m%d")
        curr_minute = datetime.now().strftime("%H%M%S")
        log_dir = f'/data/scratch/ellen660/rq-vae-transformer/tensorboard/{args.config}/{curr_time}/{curr_minute}/body_{config.arch.body.n_layer}layers_{config.arch.body.block.n_head}heads_head_{config.arch.head.n_layer}layers_{config.arch.head.block.n_head}heads_{config.arch.embd_pdrop}dropout'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        # Load the YAML file
        config = load_config("rqvae/my_code/%s.yaml" % args.config, log_dir)

    writer = init_logger(log_dir, resume)

    # torch.manual_seed(config.common.seed)
    # random.seed(config.common.seed)
    set_seed(config.common.seed)
    # data_parallel = config.distributed.data_parallel
    device = torch.device("cuda")
    # breakpoint()

    metrics_args = MetricsArgs(num_classes=config.arch.vocab_size, device=device)
    metrics = Metrics(metrics_args)

    train_loader, val_loader = init_dataset(config)
    model, model_ema = init_model(config)
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = create_optimizer(model, config)
    #simple scheduler
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.common.max_epoch)
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=config.optimizer.warmup.epoch, max_epochs=config.common.max_epoch)

    # Mixed precision scaler
    scaler = torch.amp.GradScaler(device=device)
    # optimizer = optim.AdamW(model.parameters(), lr=float(config.optimization.lr), betas=(0.8, 0.9))
    # scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=config.lr_scheduler.warmup_epoch, max_epochs=config.common.max_epoch)

    if os.path.exists(checkpoint_path):
        start_epoch = load_checkpoint(model, optimizer, scheduler, f"{checkpoint_path}/model.pth", device)
    else:
        start_epoch = 1

    # test(metrics, 0, model, val_loader, config, writer, scaler)
    for epoch in tqdm(range(start_epoch, config.common.max_epoch+1), desc="Epochs", unit="epoch"):
        train_one_step(metrics, epoch, optimizer, scheduler, model, train_loader, config, writer, scaler)
        if epoch % config.common.test_freq == 0:
            test(metrics, epoch, model, val_loader, config, writer, scaler)
        # save checkpoint and epoch
        if epoch % config.common.save_ckpt_freq == 1:
            # torch.save(model.module.state_dict(), f"{log_dir}/model.pth")
            print(f'saving')
            save_checkpoint(model, optimizer, scheduler, epoch, f"{log_dir}/model.pth")

            #TODO: set up resume