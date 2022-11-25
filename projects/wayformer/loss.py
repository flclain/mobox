import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from einops import repeat

torch.set_printoptions(precision=3, sci_mode=False)


class MyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, cls_out, reg_out, targets):
        targets, mask = targets.unbind()  # [N,T,2], [N,T]
        mask = mask.bool()
        targets[~mask] = 0  # set invalid targets to 0

        N, M, T = reg_out.shape[:3]
        mu = reg_out

        # Get closest track to gt.
        dists = (targets.unsqueeze(1)-mu).pow(2)  # [N,M,T,2]
        rep_mask = repeat(mask, "n t -> n m t 2", m=M)
        dists[~rep_mask] = 0
        dists = dists.sum([2, 3]) / rep_mask.sum([2, 3])

        dists, ids = torch.min(dists.sqrt(), dim=1)

        print(ids)
        print(torch.argmax(cls_out, dim=-1))

        # Classification loss.
        cls_loss = F.cross_entropy(cls_out, ids)

        # Regression loss.
        mu = mu[range(0, N), ids]                  # [N,T,2]

        print((mu[:, -1, :]-targets[:, -1, :]).abs())
        reg_loss = F.smooth_l1_loss(mu[mask], targets[mask])

        loss = cls_loss + reg_loss
        print(
            f"Loss: {loss.item():.3f} | cls_loss: {cls_loss.item():.3f} | reg_loss: {reg_loss.item():.3f}")
        return loss
