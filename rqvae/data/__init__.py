from .all_datasets import MergedDataset
from .datasets import AllCodes
from torch.utils.data import DataLoader
import torch

#todo: 
#config.dataset.cv
#config.dataset.max_length, depends on the compression 
#dataset weights, just shhs2 for now
#thorax, abdominal
#config.dataset.path

def init_dataset(config, ddp=False):
    cv = config.dataset.cv
    max_length = config.dataset.max_length
    root = config.dataset.path
    weights = {
        "mgh": config.dataset.mgh,
        "shhs2": config.dataset.shhs2,
        "shhs1": config.dataset.shhs1,
        "mros1": config.dataset.mros1,
        "mros2": config.dataset.mros2,
        "wsc": config.dataset.wsc,
        "cfs": config.dataset.cfs,
        "bwh": config.dataset.bwh,
    }

    train_datasets, val_datasets, train_weight, val_weight = [], [], [], []
    channels = {'thorax': config.dataset.thorax, 'abdominal': config.dataset.abdominal}
    for ds_name, weight in weights.items():
        if ds_name == "bwh":
            channels = {"thorax": 1.0}
        if weight > 0:
            train_datasets.append(AllCodes(root, dataset = ds_name, mode = "train", cv = cv, channels = channels, max_length = max_length, masking = config.dataset.masking_ratio, vocab_size = config.arch.vocab_size-1))
            val_datasets.append(AllCodes(root, dataset = ds_name, mode = "val", cv = cv, channels = channels, max_length = max_length, masking = config.dataset.masking_ratio, vocab_size = config.arch.vocab_size-1))
            train_weight.append(float(weight))
            val_weight.append(float(weight))

    #Holdout/external dataset
    val_datasets.append(AllCodes(root, dataset = config.dataset.external, mode = "val", cv = cv, channels = channels, max_length = max_length, masking = config.dataset.masking_ratio, vocab_size = config.arch.vocab_size-1))
    val_weight.append(1.)

    # merge the datasets
    train_dataset = MergedDataset(train_datasets, train_weight, 1., config.common.debug)
    val_dataset = MergedDataset(val_datasets, val_weight, 0.2, config.common.debug)

    if not ddp:
        train_loader = DataLoader(train_dataset, batch_size=config.dataset.batch_size, shuffle=True, num_workers=config.common.num_workers)
        val_loader = DataLoader(val_dataset, batch_size=config.dataset.batch_size, shuffle=False, num_workers=config.common.num_workers)
        return train_loader, val_loader, train_dataset.mapping, val_dataset.mapping
    else:
        # Use a DistributedSampler, assumes rank has been set up 
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True, drop_last=True)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.dataset.batch_size, sampler=train_sampler, num_workers=config.common.num_workers, pin_memory=True)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.dataset.batch_size, shuffle=False, sampler=val_sampler, num_workers=config.common.num_workers, pin_memory=True)
        return train_loader, val_loader, train_dataset.mapping, val_dataset.mapping