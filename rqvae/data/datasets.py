import os
import sys
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import zoom
from tqdm import tqdm
import ast

#load the bad files
bad_files = {}
with open('/data/scratch/ellen660/rq-vae-transformer/rqvae/data/bad_files.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        key, value = line.strip().split(': ')
        bad_files[key] = ast.literal_eval(value)

class AllCodes(Dataset):
    NumCv = 4
    modes = ["train", "val", "test"]
    datasets = ["bwh",  "cfs",  "mesa", "mgh", "mros1",  "mros2",  "shhs1",  "shhs2",  "wsc"]
    channels = ['thorax', 'abdominal', 'rf']
        
    def __init__(self, root, dataset = "shhs2", mode = "train", cv = 0, channels = {"thorax": 1.0}, max_length = 2 * 60 * 7, masking = 0.50, vocab_size = 512):
        assert mode in self.modes, 'Only support train, val, or test mode'
        assert dataset in self.datasets, f'Invalid dataset {dataset}'
        assert all([channel in self.channels for channel in channels.keys()]), f'Invalid channels {channels}'
        assert 0 < masking < 1, f'Masking should be between 0 and 1, got {masking}'

        self.dataset = dataset
        self.mode = mode
        self.cv = cv
        self.channels = channels
        self.ds_dir = os.path.join(root, dataset)
        self.max_length = max_length
        self.masking = masking
        self.vocab_size = vocab_size 

        # dataset preparation (only select the intersection between all channels)
        file_list = set()
        for channel in self.channels.keys():
            file_list_before = sorted([f for f in os.listdir(os.path.join(self.ds_dir, channel)) if f.endswith('.npz')])
            file_list_after = [f for f in file_list_before if f not in bad_files[dataset]]
            file_list.update(file_list_after)
        
        file_list = sorted(file_list)

        train_list, val_list = self.split_train_test(file_list)

        if mode == "train":
            self.file_list = train_list
        elif mode == "val":
            self.file_list = val_list
        elif mode == "test": #All the files
            self.file_list = file_list

    def split_train_test(self, file_list):
        train_files = []
        test_files = []
        for i in range(len(file_list)):
            if i % self.NumCv == self.cv:
                test_files.append(file_list[i])
            else:
                train_files.append(file_list[i])

        return train_files, test_files

    def __len__(self):
        return len(self.file_list)
    
    def mask_input(self, input_tokens):
        """
        Bert masking across time dimension
        """
        #input shape is (D,T)
        D, T = input_tokens.shape

        # Define the special [MASK] token
        MASK_TOKEN = self.vocab_size   

        # Create a mask
        mask = torch.rand(input_tokens.shape[1]) < self.masking 

        masked_input = input_tokens.clone()
        masked_input[:, mask] = MASK_TOKEN  # Replace with [MASK] token

        # random_tokens = torch.randint(0, self.vocab_size, input_tokens.shape)
        # replace_with_mask = torch.rand(1, T) < 0.8  # Shape (1, T)
        # replace_with_random = torch.rand(1, T) < 0.5  # Shape (1, T) #so remaining 10% of the time, we keep the original token

        # # Efficient masking using broadcasting
        # masked_input[:, mask] = torch.where(
        #     replace_with_mask[:, mask], 
        #     MASK_TOKEN, 
        #     torch.where(
        #         replace_with_random[:, mask], 
        #         random_tokens[:, mask], 
        #         input_tokens[:, mask]
        #     )
        # )
        return mask, masked_input
    
    def __getitem__(self, idx):
        filename = self.file_list[idx]

        # now randomly select a channel, sampling based on their weights
        selected_channel = np.random.choice(list(self.channels.keys()), p=list(self.channels.values()))
        filepath = os.path.join(self.ds_dir, selected_channel, filename)
        codes = np.load(filepath)['data'].squeeze()
        fs = np.load(filepath)['fs']

        if self.mode == "train":
            codes_length = codes.shape[1] - self.max_length
            #randomly sample start index
            try:
                start_idx = np.random.randint(0, codes_length+1)
            except:
                print("codes_length is negative")
                print(f"codes_length: {codes_length}")
                print("filename: ", filename)
                print(f"dataset: {self.dataset}")
                sys.exit()
            codes = codes[:, start_idx:start_idx+self.max_length]
        elif self.mode == "val":
            codes = codes[:, :self.max_length]
        elif self.mode == "test":
            codes = codes

        codes = torch.tensor(codes, dtype=torch.long)
        assert codes.max() < self.vocab_size, f"codes max {codes.max()} is greater than vocab size {self.vocab_size}"
        mask, masked_input = self.mask_input(codes)
        codes = codes.permute(1, 0)  # Swaps D and T -> New shape: (B, T, D)
        masked_input = masked_input.permute(1, 0)  
        # if self.soft:
        #     soft_codes = torch.tensor(soft_codes, dtype=torch.float32)
        #     soft_codes = soft_codes.permute(1, 0, 2)
            
        item = {
            "x": None,
            "masked_input": masked_input,
            "mask": mask,
            "filename": filename,
            "selected_channel": selected_channel
        }

        # if there is any nan or inf in the signal, return None
        if torch.isnan(codes).any() or torch.isinf(codes).any():
            # return None, 0
            print(f'bad file {filename}')
            sys.exit()

        item["x"] = codes
        return item

def main():
    dictionary = {}
    for dataset in datasets:
        print(f'dataset {dataset}')
        dictionary[dataset] = AllCodes(dataset, "train")
    
    for dataset, data in dictionary.items():
        dataloader = DataLoader(data, batch_size=8, num_workers = 10, shuffle=True)
        print(f'size of {dataset} is {len(data)}')
        for i, item in enumerate(dataloader):
            if i<=1:
                print(item["x"].shape)

if __name__ == '__main__':
    main()

