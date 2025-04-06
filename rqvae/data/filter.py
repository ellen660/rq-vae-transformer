import os
import numpy as np

root = '/data/scratch/ellen660/encodec/encodec/predictions/no_discrim/30_seconds/20250402/test'
datasets = ["shhs1", "shhs2"]
max_length = 840
bad_files = {}

for dataset in datasets:
    bad = []
    files = [f for f in os.listdir(os.path.join(root, dataset, "thorax")) if f.endswith('.npz')]
    for file in files:
        filepath = os.path.join(root, dataset, "thorax", file)
        codes = np.load(filepath)['data'].squeeze()
        if codes.shape[1] < 840:
            bad.append(file)
    bad_files[dataset] = bad

    print(f"Found {len(bad)} bad files in {dataset}")

#save the bad files for loading later
with open('/data/scratch/ellen660/rq-vae-transformer/rqvae/data/bad_files.txt', 'w') as f:
    for key, value in bad_files.items():
        f.write(f"{key}: {value}\n")

#load the bad files
with open('/data/scratch/ellen660/rq-vae-transformer/rqvae/data/bad_files.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        key, value = line.strip().split(': ')
        bad_files[key] = value.split(', ')