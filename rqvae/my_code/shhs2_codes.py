#data loader for shhs2 breathing dataset
#return the raw breathing, support deubgging

import os
import sys
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import zoom
# from .fns_to_ignore_bwh import fns_to_ignore
from tqdm import tqdm
import torch
import torch.nn.functional as F

"""
N = 2650
Number of Female: 1425
Number of male: 1225
We also filter out nights where the breathing signal is distorted or nonexistent, 
"""
fns_to_ignore_out = []

class Shhs2Dataset(Dataset):
    root = f'/data/scratch/ellen660/encodec/encodec/predictions/142145/shhs2'
    NumCv = 4
        
    def __init__(self, mode = "train", cv = 0, max_length = 2 * 60 * 7, soft=False): #1 sample every 30 seconds -> 120 samplse/hour
        self.channels = {"thorax": 1.0}
        self.mode = mode
        assert self.mode in ['train', 'test', 'val'], 'Only support train, val or test mode'
        self.cv = cv
        self.ds_dir = self.root
        self.max_length = max_length
        self.soft = soft

        # dataset preparation (only select the intersection between all channels)
        file_list = set()
        for channel in self.channels.keys():
            file_list_before = sorted([f for f in os.listdir(os.path.join(self.ds_dir, channel)) if f.endswith('.npz')])
            file_list_after = [f for f in file_list_before if f not in fns_to_ignore_out]
            file_list.update(file_list_after)

        file_list = sorted(file_list)
        new_file_list = []
        fns_to_ignore = []
        for file in file_list:
            filepath = os.path.join(self.ds_dir, "thorax", file)
            codes = np.load(filepath)['data'].squeeze()
            if codes.shape[1] > max_length:
                new_file_list.append(file)
            else:
                fns_to_ignore.append(file)
        file_list = new_file_list
        # breakpoint()
        
        self.df = pd.read_csv('/data/netmit/wifall/ADetect/data/csv/shhs2-dataset-augmented.csv')
        self.df.set_index('nsrrid', inplace=True)
        #get labels of patients with OH
        self.female, self.male = self.get_patients(file_list)

        train_list, val_list = self.split_train_test(self.female, self.male)
            
        if mode == "train":
            self.file_list = train_list
        elif mode == "val":
            self.file_list = val_list
        elif mode == "test": #All the files
            self.file_list = file_list
        else:
            raise ValueError(f"Invalid mode: {mode}")
        
    def get_patients(self, file_list):
        """
        Returns a list of patients with orthostatic hypotension and without
        """
        female = []
        male = []
        for idx in range(len(file_list)):
            if self.get_gender_label(idx, file_list) == 1:
                female.append(file_list[idx])
            else:
                male.append(file_list[idx])
        return female, male

    def split_train_test(self, female, male):
        train_files = []
        val_files = []
        for i in range(len(female)):
            if i % self.NumCv == self.cv:
                val_files.append(female[i])
            else:
                train_files.append(female[i])
        for i in range(len(male)):
            if i % self.NumCv == self.cv:
                val_files.append(male[i])
            else:
                train_files.append(male[i])
        
        return train_files, val_files

    def __len__(self):
        return len(self.file_list)
    
    # def process_signal(self, signal, fs):
    #     assert fs == 200, f"fs is not 200 but {fs}"
    #     signal, _, _ = detect_motion_iterative(signal, fs)
    #     signal = signal_crop(signal)
    #     signal = norm_sig(signal)

    #     if fs != 10:
    #         signal = zoom(signal, 10/fs)
    #         fs = 10

    #     return signal
        
    def get_gender_label(self, idx, file_list=None):
        """
        Leaving here for now because no labels yet
        filename: shhs2-200077.npz
        label: mit_gender or rawbp_s2
        For gender, 0 is male
        1 is female
        """
        if file_list:
            filename=file_list[idx]
        else:
            filename = self.file_list[idx]
        nsrrid = filename.split('-')[1].split('.')[0]
        gender = self.df.loc[int(nsrrid)]["mit_gender"] #labelled as 1 or 2 
        gender -= 1 #make gender 0 or 1 instead
        return torch.tensor(gender, dtype=torch.float32)

    def __getitem__(self, idx):
        filename = self.file_list[idx]

        # now randomly select a channel, sampling based on their weights
        selected_channel = np.random.choice(list(self.channels.keys()), p=list(self.channels.values()))
        filepath = os.path.join(self.ds_dir, selected_channel, filename)
        codes = np.load(filepath)['data'].squeeze()
        if self.soft:
            soft_codes = np.load(filepath)['soft_targets'].squeeze()
        fs = np.load(filepath)['fs']
        # print(f'codes shape: {codes.shape}, fs: {fs}, soft {soft_codes.shape}')
        assert (fs - 0.0333333 < 1e-4) , "Sampling rate is not 0.0333Hz"

        if self.mode == "train":
            codes_length = codes.shape[1] - self.max_length
            #randomly sample start index
            try:
                start_idx = np.random.randint(0, codes_length)
            except:
                print("codes_length is negative")
                print(f"codes_length: {codes_length}")
                print("filename: ", filename)
                # print(f"start_idx: {start_idx}")
                sys.exit()
            codes = codes[:, start_idx:start_idx+self.max_length]
            if self.soft:
                soft_codes = soft_codes[:, start_idx:start_idx+self.max_length, :]
        elif self.mode == "val":
            codes = codes[:, :self.max_length]
            if self.soft:
                soft_codes = soft_codes[:, :self.max_length, :]
        elif self.mode == "test":
            # codes = codes
            codes_length = codes.shape[1] - self.max_length
            #randomly sample start index
            try:
                start_idx = np.random.randint(0, codes_length)
            except:
                print("codes_length is negative")
                print(f"codes_length: {codes_length}")
                print("filename: ", filename)
                # print(f"start_idx: {start_idx}")
                sys.exit()
            codes = codes[:, start_idx:start_idx+self.max_length]
            if self.soft:
                soft_codes = soft_codes[:, start_idx:start_idx+self.max_length, :]
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        codes = torch.tensor(codes, dtype=torch.int)
        codes = codes.permute(1, 0)  # Swaps D and T -> New shape: (B, T, D)
        if self.soft:
            soft_codes = torch.tensor(soft_codes, dtype=torch.float32)
            soft_codes = soft_codes.permute(1, 0, 2)
        
        gender = self.get_gender_label(idx)

        item = {
            "x": None,
            # "y": soft_codes,
            "gender": gender,
            "filename": filename,
            "selected_channel": selected_channel
        }

        # if there is any nan or inf in the signal, return None
        if torch.isnan(codes).any() or torch.isinf(codes).any():
            # return None, 0
            print(f'bad file {filename}')
            return item

        #unsquzze dim0
        # codes = codes.unsqueeze(0)
        item["x"] = codes
        if self.soft:
            item["y"] = soft_codes
        else:
            item["y"] = torch.zeros(1)

        return item

def main():
    dataset = Shhs2Dataset(mode="val")
    print(f"Dataset size is {len(dataset)}")
    dataloader = DataLoader(dataset, batch_size=8, num_workers = 4, shuffle=True)
    num_OH, control = 0, 0

    for i, item in enumerate(dataloader):
        codes = item["x"][0]
        if i <= 5:
            print(f'codes {codes.shape}')
        y = item["gender"]
        # if y == 1:
        #     num_OH += 1
        # else:
        #     control += 1
    print(f'num OH {num_OH}')
    print(f'num control {control}')

if __name__ == '__main__':
    main()
