import torch.nn.functional as F
import torchmetrics
import sys
import torch

@torch.no_grad()
def compute_loss(logits, targets, soft=True, mask=None):
    if not soft:
        targets = targets.long()
    B, T, D, N = logits.shape

    #mask shape is 840

    metrics = {}
    auroc = torchmetrics.AUROC(task="multiclass", num_classes=512).to(logits.device)
    top1_acc = torchmetrics.Accuracy(task="multiclass", num_classes=N, top_k=1).to(logits.device)
    top5_acc = torchmetrics.Accuracy(task="multiclass", num_classes=N, top_k=5).to(logits.device)
    auroc_per_depth = []
    top_1 = []
    top_5 = []

    #per layer cross entropy
    for d in range(D):
        # Select the logits for this depth and reshape to (B*T, N)
        logit_d = logits[:, :, d, :].reshape(B * T, N)
        if soft:
            target_d = targets[:, :, d, :].reshape(B * T, N)
        else:
            target_d = targets[:, :, d].reshape(B * T)  # (B*T,)
        
        # Compute loss
        loss = F.cross_entropy(logit_d, target_d)
        metrics[f'Cross Entropy Codebook {d}'] = loss

        # Compute AUROC
        # auroc_d = auroc(logit_d, target_d)  # Assuming self.auroc handles multi-class probs
        # auroc_per_depth.append(auroc_d)
        # metrics[f'AUROC Codebook {d}'] = auroc_d
        
        #compute top-1, top-5
        top1 = top1_acc(logit_d, target_d)
        top5 = top5_acc(logit_d, target_d)
        metrics[f'Top 1 {d}'] = top1
        metrics[f'Top 5 {d}'] = top5
        top_1.append(top1)
        top_5.append(top5)
    
    # Store AUROC
    # metrics["auroc"] = sum(auroc_per_depth) / D  # Macro AUROC
    metrics["top1"] = sum(top_1) / D
    metrics["top5"] = sum(top_5) / D

    return metrics