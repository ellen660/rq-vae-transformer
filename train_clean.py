import os
import argparse
import math
import time
import sys

import rqvae.utils.dist as dist_utils
from rqvae.models import create_model
from rqvae.models.rqtransformer.configs import RQTransformerConfig, AttentionStackConfig, AttentionBlockConfig
from rqvae.data import init_dataset
from rqvae.losses import compute_loss, Metrics, MetricsArgs, LinearWarmupCosineAnnealingLR
from rqvae.optimizer import create_optimizer, create_scheduler
from rqvae.utils.utils import set_seed, compute_model_size, get_num_conv_linear_layers

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from datetime import datetime
import yaml
from tqdm import tqdm
import argparse
from dataclasses import fields, is_dataclass
import torch.utils.checkpoint as checkpoint
import functools

def predict_future(model, logits, tau=0.1):
    soft_tokens = F.gumbel_softmax(logits[:, 1:, :, :], tau, hard=False)
    hard_tokens = torch.argmax(logits[:, 1:, :, :], dim=-1)
    ste_tokens = (F.one_hot(hard_tokens.to(torch.int64), config.arch.vocab_size-1).float() - soft_tokens).detach() + soft_tokens
    logits2 = model(xs=ste_tokens, amp=config.common.amp, one_hot=True)
    return logits2

def train_one_step(metrics, epoch, optimizer, scheduler, model, train_loader, config, writer, scaler, label_mapping):
    model.train()
    epoch_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch}", unit="batch")

    for i, (item, ds_id) in enumerate(progress_bar):
        x = item["masked_input"].to(device)
        target = item["x"].to(device)
        mask = item["mask"].unsqueeze(-1).expand(-1, -1, config.arch.block_size[1])
        mask = mask.to(device)
        # print(f'x shape: {x.shape} {x.dtype}') #B, T, D
        # print(f'mask shape: {mask.shape} {mask.dtype}') #B, T, D
        loss = 0
        
        #First pass
        logits = model(xs=x, amp=config.common.amp)  #B, T, D, vocab_size
        # print(f'logits {logits.shape}') #B, T, D, vocab_size
        loss += model.module.compute_loss(logits, target, use_soft_target=config.loss.soft, mask=mask)

        #Predict future steps

        # for i in range(config.arch.num_steps-1):
            # logits_i = predict_future(model, logits, tau=0.1)
            # loss += model.module.compute_loss(logits_i, target[:, i+1:, :], use_soft_target=config.loss.soft)
            # logits = logits_i
            
        # Predict two steps ahead
        # logits2 = predict_future(model, logits, tau=0.1)
        # loss += model.module.compute_loss(logits2, target[:, 2:, :], use_soft_target=config.loss.soft)

        # # Predict three steps ahead
        # logits3 = predict_future(model, logits2, tau=0.1)
        # loss += model.module.compute_loss(logits3, target[:, 3:, :], use_soft_target=config.loss.soft)

        # # Predict four steps ahead
        # logits4 = predict_future(model, logits3, tau=0.1)
        # loss += model.module.compute_loss(logits4, target[:, 4:, :], use_soft_target=config.loss.soft)
            
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
        losses = compute_loss(logits, target, soft=False, ds_id=ds_id, mask=mask)
        for d_id in losses.keys():
            metrics[d_id].fill_metrics(losses[d_id], epoch*len(train_loader) + i)

    scheduler.step()  # Update learning rate
    loss_per_epoch = epoch_loss/len(train_loader)
    print(f"Epoch {epoch}, training loss: {loss_per_epoch}")
    for d_id in label_mapping.keys():
        metrics_dict = metrics[d_id].compute_and_log_metrics(loss_per_epoch)
        # log the learning rate
        metrics_dict['Learning Rate'] = optimizer.param_groups[0]['lr']
        logger(writer, metrics_dict, 'train', label_mapping[d_id], epoch)
        metrics[d_id].clear_metrics()

@torch.no_grad()
def test(metrics, epoch, model, val_loader, config, writer, scaler, label_mapping):
    model.eval()
    epoch_loss = 0

    progress_bar = tqdm(val_loader, desc=f"Validation Epoch {epoch}", unit="batch")

    for i, (item, ds_id) in enumerate(progress_bar):
        x = item["masked_input"].to(device)
        target = item["x"].to(device)
        mask = item["mask"].unsqueeze(-1).expand(-1, -1, config.arch.block_size[1]).to(device)

        logits = model(x)  # Forward pass
        loss = model.module.compute_loss(logits, target, use_soft_target=config.loss.soft, mask=mask)
        
        epoch_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())
        losses = compute_loss(logits, target, soft=False, ds_id=ds_id, mask=mask)
        for d_id in losses.keys():
            metrics[d_id].fill_metrics(losses[d_id], epoch*len(val_loader) + i)

    loss_per_epoch = epoch_loss/len(val_loader)
    print(f"Epoch {epoch}, validation loss: {loss_per_epoch}")
    for d_id in label_mapping.keys():
        metrics_dict = metrics[d_id].compute_and_log_metrics(loss_per_epoch)
        # log the learning rate
        metrics_dict['Learning Rate'] = optimizer.param_groups[0]['lr']
        logger(writer, metrics_dict, 'val', label_mapping[d_id], epoch)
        metrics[d_id].clear_metrics()

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
def logger(writer, metrics, phase, ds_name, epoch_index):
    for key, value in metrics.items():
        if type(value)!= float and len(value.shape) > 0 and value.shape[0] == 2:
            value = value[1]
        elif type(value)!= float and len(value.shape) > 0 and value.shape[0] > 2:
            raise Exception("Need to handle multiclass")
            # bp()
        writer.add_scalar("%s/%s/%s"%(ds_name, phase, key), value, epoch_index)
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
    
    parser.add_argument("--config", type=str, default="mlm")
    parser.add_argument("--resume_from", type=str, default="")
    parser.add_argument("--log_dir", type=str, default=None)

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
        config = load_config("rqvae/params/%s.yaml" % args.config)
        if args.log_dir:
            log_dir = args.log_dir
        else:
            curr_time, curr_min = time.strftime("%Y-%m-%d_%H-%M", time.localtime()).split("_")
            log_dir = f'/data/scratch/ellen660/rq-vae-transformer/tensorboard/{config.exp_details.name}/{config.exp_details.description}/{curr_time}_{curr_min}/body_{config.arch.body.n_layer}layers_{config.arch.body.block.n_head}heads_head_{config.arch.head.n_layer}layers_{config.arch.head.block.n_head}heads_{config.arch.embd_pdrop}dropout'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        # Load the YAML file
        config = load_config("rqvae/params/%s.yaml" % args.config, log_dir)

    set_seed(config.common.seed)
    device = torch.device("cuda")
    train_loader, val_loader, train_mapping, val_mapping = init_dataset(config, ddp=False)
    model, model_ema = init_model(config)
    model = model.to(device)
    if not config.common.distributed:
        model = nn.DataParallel(model)

    optimizer = create_optimizer(model, config)
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=config.optimizer.warmup.epoch, max_epochs=config.common.max_epoch)
    # Mixed precision scaler
    scaler = torch.amp.GradScaler(device=device)
    # optimizer = optim.AdamW(model.parameters(), lr=float(config.optimization.lr), betas=(0.8, 0.9))

    writer = init_logger(log_dir, resume)
    metrics = {}
    for label in val_mapping.keys():
        metrics_args = MetricsArgs(num_classes=config.arch.vocab_size, device=device)
        metrics[label] = Metrics(metrics_args)

    if os.path.exists(checkpoint_path):
        start_epoch = load_checkpoint(model, optimizer, scheduler, f"{checkpoint_path}/model.pth", device)
    else:
        start_epoch = 1

    # test(metrics, start_epoch, model, val_loader, config, writer, scaler, val_mapping)
    for epoch in tqdm(range(start_epoch, config.common.max_epoch+2), desc="Epochs", unit="epoch"):
        train_one_step(metrics, epoch, optimizer, scheduler, model, train_loader, config, writer, scaler, train_mapping)
        if epoch % config.common.test_every == 0:
            test(metrics, epoch, model, val_loader, config, writer, scaler, val_mapping)
        # save checkpoint and epoch
        if epoch % config.common.save_every == 1:
            save_checkpoint(model, optimizer, scheduler, epoch, f"{log_dir}/model.pth")


