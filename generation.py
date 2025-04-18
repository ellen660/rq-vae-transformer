import os
import argparse
import math
import sys

import rqvae.utils.dist as dist_utils
from rqvae.models import create_model
from rqvae.models.rqtransformer.configs import RQTransformerConfig, AttentionStackConfig, AttentionBlockConfig
from rqvae.losses import compute_loss, Metrics, MetricsArgs, LinearWarmupCosineAnnealingLR
from rqvae.data import test_dataset
from rqvae.optimizer import create_optimizer, create_scheduler
from rqvae.utils.utils import set_seed, compute_model_size, get_num_conv_linear_layers
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
from dataclasses import fields, is_dataclass

# Get the absolute path of the 'encoded' folder
encoded_path = os.path.abspath("../encodec")  # Adjust path if necessary

# Add it to sys.path
sys.path.append(encoded_path)

# Now import the model
# from encodec import EncodecModel
# from encodec.my_code.spectrogram_loss import ReconstructionLoss, ReconstructionLosses

@torch.no_grad()
def plot_attention_matrix(model, val_loader, config, save_dir):
    model.eval()
    progress_bar = tqdm(val_loader, desc=f"Visualize Attention", unit="batch")
    matrices = {}
    for i, item in enumerate(progress_bar):
        if i>=10:
            break
        x = item["x"]
        x = x.to(device)

        save_path = f'{save_dir}/shhs2/{item["filename"][0]}_attention.pdf'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        attention_matrix = model.get_attention_matrix(x)
        fig, axs = plt.subplots(4, 3, figsize=(20, 20), sharex=False)
        axs = axs.flatten()
        for i, layer in enumerate(attention_matrix):
            # breakpoint()
            attn_weights = layer[0].detach().cpu().numpy()
            # attn_weights = (attn_weights - np.min(attn_weights)) / (np.max(attn_weights) - np.min(attn_weights))
            # attn_weights = np.log1p(attn_weights)
            axs[i].pcolormesh(attn_weights, cmap='magma', vmin=0, vmax=0.2)#0th head
            axs[i].set_title(f'Layer {i} Head 0'), #vmin=0, vmax=0.2, subtravt one and two 
        
        fig.tight_layout()
        fig.savefig(f'{save_dir}/shhs2/{item["filename"][0]}_attention.png')
        # plt.savefig(save_path, format="pdf", bbox_inches="tight")
        plt.close(fig)

        # breakpoint()
        matrices[item["filename"][0]] = attention_matrix
    return matrices

@torch.no_grad()
def test(model, rvqvae, val_loader, config, save_dir):
    model.eval()
    rvqvae.eval()

    progress_bar = tqdm(val_loader, desc=f"Generation", unit="batch")

    for i, (item) in enumerate(progress_bar):
        if i>=10:
            break
        x = item["x"]
        x = x.to(device)

        logits = model(x)  # Forward pass
        code_predictions = logits.argmax(dim=-1) #B, T, D

        code_predictions = code_predictions.permute(2, 0, 1)
        x = x.permute(2, 0, 1)
        quantized_prediction = rvqvae.quantizer.decode(code_predictions) #D, B, T
        output_prediction = rvqvae.decoder(quantized_prediction)
        quantized_actual = rvqvae.quantizer.decode(x)
        output_actual = rvqvae.decoder(quantized_actual)
        diff = output_actual - output_prediction
        fig, axs = plt.subplots(4, 1, figsize=(20, 10), sharex=False)
        print(f'shape {output_actual.shape}')

        freq_loss_dict = freq_loss(output_actual, output_prediction)
        S_x = freq_loss_dict["S_x"]
        S_x_hat = freq_loss_dict["S_x_hat"]
        _, num_freq, _ = S_x.size()
        S_x = S_x[:, :num_freq//2, :]
        S_x_hat = S_x_hat[:, :num_freq//2, :]

        # use this to set the scale of the spectrogram
        min_spec_val = min(S_x.min(), S_x_hat.min())
        max_spec_val = max(S_x.max(), S_x_hat.max())
        # breakpoint()

        time_start = 0
        # time_end = output_actual.shape[0]
        time_end = 1200

        x_time = np.arange(time_start, time_end, 1)
        print(f'time shape {x_time.shape}')

        axs[0].plot(x_time, output_prediction.detach().cpu().numpy().squeeze()[10000:10000+10*120])
        axs[0].set_title('Reconstructed RVQVAE')
        axs[0].set_ylim(-6, 6)

        axs[1].plot(x_time, output_actual.detach().cpu().numpy().squeeze()[10000:10000+10*120])
        axs[1].set_title('Generated Transformer')
        axs[1].set_ylim(-6, 6)

        # axs[2].plot(x_time[10000:10000+10*120], diff[10000:10000+10*120])
        # axs[2].set_title('Differenec')
        # axs[2].set_ylim(-6, 6)

        # axs1[0].plot(x_time, x[0].cpu().numpy().squeeze())
        # axs1[0].set_title('Original')
        # axs1[0].set_ylim(-6, 6)
        axs[2].imshow(S_x.detach().cpu().numpy()[0], cmap='jet', aspect='auto', extent=[time_start, time_end, 0, num_freq//2], vmin=min_spec_val, vmax=max_spec_val)
        axs[2].invert_yaxis()
        axs[2].set_title('Reconstructed Spectrogram')

        axs[3].imshow(S_x_hat.detach().cpu().numpy()[0], cmap='jet', aspect='auto', extent=[time_start, time_end, 0, num_freq//2], vmin=min_spec_val, vmax=max_spec_val)
        axs[3].invert_yaxis()
        axs[3].set_title('Transformer Spectrogram L1 loss {:.6f}'.format(freq_loss_dict["l1_loss"]))

        #save fig 
        fig.tight_layout()
        fig.savefig(f'{save_dir}/shhs2/{item["filename"][0]}.png')
        plt.show()
        # breakpoint()

def generate_embeddings(model, ds_name, data_loader, save_dir):
    model.eval()
    # rvqvae.eval()

    progress_bar = tqdm(data_loader, desc=f"Generation", unit="batch")

    for i, (item) in enumerate(progress_bar):
        x = item["x"]
        x = x.to(device)

        embeddings = model(x, return_embeddings=True)  # Forward pass
        # print(f'embedding shape {embeddings.shape}')
        # x = x.permute(2, 0, 1)
        # quantized_actual = rvqvae.quantizer.decode(x) #D, B, T
        # quantized_actual = quantized_actual.permute(0, 2, 1)
        # print(f'rvqvae shape {quantized_actual.shape}')
        # quantized=quantized_actual.squeeze().cpu().detach().numpy(), 
        # print(f'x shape {x.shape}')

        # Save the codes
        save_path = os.path.join(save_dir, ds_name, item["filename"][0])
        np.savez(save_path, transformer=embeddings.squeeze().cpu().detach().numpy())
        # breakpoint()

class ConfigNamespace:
    """Converts a dictionary into an object-like namespace for easy attribute access."""
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                value = ConfigNamespace(value)  # Recursively convert nested dictionaries
            setattr(self, key, value)

def load_config(filepath):
    #make directory
    with open(filepath, "r") as file:
        config_dict = yaml.safe_load(file)
    return ConfigNamespace(config_dict)

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

# Load the YAML file and convert to ConfigNamespace
def load_rqvae_config(filepath, log_dir=None):
    #make directory
    with open(filepath, "r") as file:
        config_dict = yaml.safe_load(file)
    return ConfigNamespace(config_dict)

def init_rqvae_model(config):
    model = EncodecModel._get_model(
        config.model.target_bandwidths, 
        config.model.sample_rate, 
        config.model.channels,
        causal=config.model.causal, model_norm=config.model.norm, 
        audio_normalize=config.model.audio_normalize,
        segment=eval(config.model.segment), name=config.model.name,
        ratios=config.model.ratios,
        bins=config.model.bins,
        dimension=config.model.dimension,
    )
    # disc_model = MultiScaleSTFTDiscriminator(
    #     in_channels=config.model.channels,
    #     out_channels=config.model.channels,
    #     filters=config.model.filters,
    #     hop_lengths=config.model.disc_hop_lengths,
    #     win_lengths=config.model.disc_win_lengths,
    #     n_ffts=config.model.disc_n_ffts,
    # )

    # log model, disc model parameters and train mode
    # print(model)
    # print(disc_model)
    print(f"model train mode :{model.training} | quantizer train mode :{model.quantizer.training} ")
    # print(f"disc model train mode :{disc_model.training}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model Total number of parameters: {total_params}")
    # total_params = sum(p.numel() for p in disc_model.parameters())
    # print(f"Discriminator Total number of parameters: {total_params}")
    return model

def set_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--config", type=str, default="test")
    # parser.add_argument("--model_path", type=str, default=f"mlm/30_seconds/20250409-0153/ max_epoch=400 masking_ratio=0.75 batch_size=6 init_lr=0.0005") #1step
    # parser.add_argument("--model_path", type=str, default=f"/data/scratch/ellen660/rq-vae-transformer/tensorboard/test/20250227/220723/body_12layers_16heads_head_12layers_16heads_0.1dropout")# 4step
    # parser.add_argument("--model_path", type=str, default=f"/data/scratch/ellen660/rq-vae-transformer/tensorboard/mlm/20250306/093732/body_12layers_16heads_head_12layers_16heads_0.0dropout")
    return parser.parse_args()

if __name__ == "__main__":
    # args = set_args()
    model_paths ={
        # "Autoregressive 30 seconds 1 step new": "autoregressive/30_seconds/2025-04-06_15-13/body_12layers_16heads_head_12layers_16heads_0.1dropout",
        # "Autoregressive 30 seconds 4 step old": "test/20250227/220723/body_12layers_16heads_head_12layers_16heads_0.1dropout",
        # "Autoregressive 30 seconds 1 step old": "test/20250218/body_12layers_16heads_head_12layers_16heads",
        # "Autoregressive stacking 6 seconds 1 step reduced dim": "autoregressive_stacked/6_second_6_codebooks/20250409-1604/ max_epoch=400 batch_size=4 init_lr=0.0005 num_steps=1 stack_every=5",
        # "Autoregressive stacking 6 seconds 4 step reduced dim": "autoregressive_stacked/6_second_6_codebooks/20250409-1605/ max_epoch=400 batch_size=4 init_lr=0.0005 num_steps=4 stack_every=5",
        # "Autoregressive no stacking 6 seconds 1 step reduced dim": "autoregressive_stacked/6_second_6_codebooks/20250412-2334/ max_epoch=400 batch_size=6 init_lr=0.0005 num_steps=1 stack_every=1",
        "Autoregressive no stacking 6 seconds 4 step reduced dim": "autoregressive_stacked/6_second_6_codebooks/20250412-2329/ max_epoch=400 batch_size=4 init_lr=0.0005 num_steps=4 stack_every=1",
    }
    user_name = os.getlogin()

    for model_name, model_path in model_paths.items():
        save_dir = f'/data/scratch/ellen660/rq-vae-transformer/predictions'
        save_dir = os.path.join(save_dir, model_name)
        checkpoint_path = f'/data/scratch/ellen660/rq-vae-transformer/tensorboard/{model_path}'
        # checkpoint_paths = [f"/data/scratch/ellen660/rq-vae-transformer/tensorboard/test/20250218/body_12layers_16heads_head_12layers_16heads", f"/data/scratch/ellen660/rq-vae-transformer/tensorboard/test/20250227/220723/body_12layers_16heads_head_12layers_16heads_0.1dropout"]

        # Load the YAML file
        config = load_config(f"{checkpoint_path}/config.yaml")

        set_seed(config.common.seed)
        device = torch.device("cuda")
        test_datasets = test_dataset(config)
        for ds_name in test_datasets:
            os.makedirs(os.path.join(save_dir, ds_name), exist_ok=True)

        model, model_ema = init_model(config)
        model = model.to(device)

        checkpoint = torch.load(f"{checkpoint_path}/model.pth", map_location=device)
        # breakpoint()
        model.load_state_dict(checkpoint['model_state_dict'])

        # rqvae_dir = "/data/scratch/ellen660/encodec/encodec/tensorboard/091224_l1/20250209/142145"
        # rqvae_config = load_rqvae_config(f'{rqvae_dir}/config.yaml', rqvae_dir)
        # model_rqvae = init_rqvae_model(rqvae_config)
        # model_rqvae = model_rqvae.to(device)
        # checkpoint_rvqvae = torch.load(f"{rqvae_dir}/model.pth", map_location=device)
        # freq_loss = ReconstructionLoss(alpha=rqvae_config.loss.alpha, bandwidth=rqvae_config.loss.bandwidth, sampling_rate=10, n_fft=rqvae_config.loss.n_fft, device=device)

        # checkpoint_disc = torch.load(checkpoint_path_disc, map_location=device)

        # model_rqvae.load_state_dict(checkpoint_rvqvae)
        # disc.load_state_dict(checkpoint_disc)
        print("checkpoint loaded successfully")

        # test(model, model_rqvae, val_loader, config, save_dir)
        # generate_embeddings(model, model_rqvae, test_loader, config, save_dir)
        for ds_name, test_loader in test_datasets.items():
            print(f"Testing {ds_name} dataset")
            generate_embeddings(model, ds_name, test_loader, save_dir)
            # plot_most_frequent_signals(ds_name, pivot, model, save_dir, config, device)
            # matrices = plot_attention_matrix(model, test_loader, config, save_dir)
        # breakpoint()


# def plot_most_frequent_signals(ds_name, pivot, model, save_dir, config, device):
#     num_codebooks = int(100 * config.model.target_bandwidths[0])
#     # print(f'num codebooks {num_codebooks}')
#     codes = []
#     for codebook_idx in range(num_codebooks):
#         most_common_code = pivot[codebook_idx][0]
#         codes.append(most_common_code)
#     codes = torch.tensor(codes).unsqueeze(1).unsqueeze(2) #N, B, T
#     codes = codes.to(device)

#     # Plot histograms for each feature
#     fig, axes = plt.subplots(8, 4, figsize=(20, 15))  # 8 rows, 4 columns for 32 features
#     axes = axes.flatten()

#     prev = None
#     for n_q in range(1,num_codebooks+1):
#         quantized = model.quantizer.decode(codes, n_q=n_q)
#         output = model.decoder(quantized).detach().cpu().numpy().squeeze()
#         if prev is not None:
#             diff = output - prev 
#         else:
#             diff = output
#         mean = np.mean(diff)
#         std = np.std(diff)
#         time = np.arange(0, 300)
#         axes[n_q-1].plot(time, diff)
#         axes[n_q-1].set_title(f"Signal n_q={n_q} - Signal n_q={n_q-1}, mean {float(f'{mean:.6f}')}, std {float(f'{std:.6f}')}")
#         axes[n_q-1].set_ylim(-0.5, 0.5)
#         prev = output
    
#     plt.tight_layout()
#     save_path = os.path.join(save_dir, ds_name, f"{ds_name}_most_common_signals.png")  # Save as PNG
#     plt.savefig(save_path, dpi=300, bbox_inches="tight")  # High-quality save
#     plt.close()  # Close the figure to free memory

#     #plot the most common signal 
#     quantized = model.quantizer.decode(codes, n_q=num_codebooks)
#     output = model.decoder(quantized).detach().cpu().numpy().squeeze()
#     fig, ax = plt.subplots()  # Create a single axes (not multiple)
#     # Plot only the specified axes
#     time = np.arange(0, 300)
#     ax.plot(time, output)
#     ax.set_title(f"{ds_name}_generic_signal")
#     ax.set_ylim(-2, 2)
#     save_path = os.path.join(save_dir, ds_name, f"{ds_name}_generic_signal.png")  # Save as PNG
#     plt.savefig(save_path, dpi=300, bbox_inches="tight")  # High-quality save
#     plt.close()  # Close the figure to free memory

#     print(f"Finished processing {ds_name}")


    