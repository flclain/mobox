import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from einops import repeat
from mobox.losses.soft_target_cross_entropy import soft_target_cross_entropy_loss

torch.set_printoptions(precision=3, sci_mode=False)


class MyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.count = 0

    def forward(self, cls_out, reg_out, targets):
        targets, mask = targets.unbind()  # [N,T,2], [N,T]
        mask = mask.bool()
        targets[~mask] = 0

        # Get best mode.
        N, M, T = reg_out.shape[:3]
        rep_mask = repeat(mask, "N T -> N M T", M=M)
        dists = ((targets.view(N, 1, T, 2) - reg_out).norm(p=2, dim=-1) * rep_mask).sum(-1)  # [N,M]
        ids = torch.argmin(dists, dim=-1)
        print(ids)
        print(torch.argmax(cls_out, dim=-1))

        # Classification loss.
        # cls_loss = F.cross_entropy(cls_out, ids)
        soft_target = F.softmax(-dists / mask.sum(dim=-1, keepdim=True), dim=-1).detach()  # [N,M]
        # cls_loss = soft_target_cross_entropy_loss(cls_out, soft_target)
        cls_loss = F.binary_cross_entropy_with_logits(cls_out, soft_target, reduction="sum") / N

        # probs = F.softmax(cls_out, dim=-1)
        probs = F.sigmoid(cls_out)
        print(probs[range(0, N), ids])
        print(soft_target[range(0, N), ids])

        # Regression loss.
        mu = reg_out
        mu = mu[range(0, N), ids]  # [N,T,2]

        print((mu[:, -1, :]-targets[:, -1, :]).abs())

        reg_loss = F.smooth_l1_loss(mu[mask], targets[mask])
        loss = cls_loss + reg_loss

        self.count += 1
        if self.count % 100 == 0:
            plt.gca().set_aspect("equal")
            for t in targets:
                t = t.detach().numpy()
                plt.plot(t[:, 0], t[:, 1], 'y.-')

            for t in mu:
                t = t.detach().numpy()
                plt.plot(t[:, 0], t[:, 1], 'r.-')
            # plt.show()
            plt.savefig(f"{self.count}.svg")
            plt.cla()
            plt.clf()

        print(
            f"Loss: {loss.item():.3f} | cls_loss: {cls_loss.item():.3f} | reg_loss: {reg_loss.item():.3f}")
        return loss
