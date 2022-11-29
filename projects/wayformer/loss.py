import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from einops import repeat

torch.set_printoptions(precision=3, sci_mode=False)


class MyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.count = 0
        d = torch.load("./cache/anchors.pth")
        self.register_buffer("anchors", d)  # [M,T,2]

    def forward(self, cls_out, reg_out, targets):
        targets, mask = targets.unbind()  # [N,T,2], [N,T]
        mask = mask.bool()

        N, M, T = reg_out.shape[:3]
        mu = reg_out + self.anchors

        # Get closest track.
        end_points = targets[:, -1:, :]
        dists = (end_points - self.anchors[:, -1]).pow(2).sum(-1)  # [N,M]
        dists, ids = torch.min(dists, dim=1)

        print(ids)
        print(torch.argmax(cls_out, dim=-1))

        # Classification loss.
        cls_loss = F.cross_entropy(cls_out, ids)

        # Regression loss.
        mu = mu[range(0, N), ids]                  # [N,T,2]

        print((mu[:, -1, :]-targets[:, -1, :]).abs())
        reg_loss = F.smooth_l1_loss(mu[mask], targets[mask], reduction="none")
        reg_loss = reg_loss.sum() / N

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
            plt.show()

        print(
            f"Loss: {loss.item():.3f} | cls_loss: {cls_loss.item():.3f} | reg_loss: {reg_loss.item():.3f}")
        return loss
