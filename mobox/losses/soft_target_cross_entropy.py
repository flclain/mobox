"""Soft target cross entropy loss.

Reference:
  https://github.com/ZikangZhou/HiVT/blob/main/losses/soft_target_cross_entropy_loss.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_target_cross_entropy_loss(preds, targets, reduction="mean"):
    loss = torch.sum(-targets * F.log_softmax(preds, dim=-1), dim=-1)
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "none":
        return loss
    raise ValueError(f'{reduction} is not a valid value for reduction')


class SoftTargetCrossEntropyLoss(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, preds, targets):
        loss = soft_target_cross_entropy_loss(preds, targets, self.reduction)
        return loss
