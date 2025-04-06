import torchmetrics
import numpy as np 
from dataclasses import dataclass
import torch.nn.functional as F

@dataclass
class MetricsArgs():
    device: str
    num_classes: int

class Metrics():
    def __init__(self, args: MetricsArgs):
        self.args = args 
        self.used_keys = {}
        self.init_metrics()
        self.auroc = torchmetrics.AUROC(task="multiclass", num_classes=self.args.num_classes).to(self.args.device)

    def init_metrics(self):
        self.metrics_dict = {
            # "auroc": {},
            "top1": {},
            "top5":{}
        }
        for i in range(32):
            self.metrics_dict[f"Cross Entropy {i}"] = {}
            self.metrics_dict[f'Top 1 {i}'] = {}
            self.metrics_dict[f'Top 5 {i}'] = {}
            
            # self.metrics_dict[f"AUROC Codebook {i}"] = {}
        self.metrics = set(self.metrics_dict.keys())
    
    def fill_metrics(self, losses, epoch):
        for key, value in losses.items():
            self.metrics_dict[key][epoch] = value
            self.used_keys[key] = True

    def compute_and_log_metrics(self, loss=0):
        metrics = {}
        for item in self.used_keys:
            metrics[item] = sum(self.metrics_dict[item].values()) / len(self.metrics_dict[item])
        
        if loss != 0:
            metrics['Cross Entropy'] = loss

        return metrics
    
    def clear_metrics(self):
        self.metrics_dict = {
            # "auroc": {},
            "top1": {},
            "top5":{}
        }
        for i in range(32):
            self.metrics_dict[f"Cross Entropy {i}"] = {}
            # self.metrics_dict[f"AUROC Codebook {i}"] = {}
            self.metrics_dict[f'Top 1 {i}'] = {}
            self.metrics_dict[f'Top 5 {i}'] = {}
        self.used_keys = {}

