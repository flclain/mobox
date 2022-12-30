"""UST: Unifying Spatio-Temporal Context for Trajectory Prediction in Autonomous Driving"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange


class USTEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layer1 = nn.Linear(in_channels, 64)
        self.ln1 = nn.LayerNorm(64)

        self.layer2 = nn.Linear(64, 64)
        self.ln2 = nn.LayerNorm(64)

        self.layer3 = nn.Linear(64, 128)
        self.ln3 = nn.LayerNorm(128)

        self.layer4 = nn.Linear(128+in_channels, 128)
        self.ln4 = nn.LayerNorm(128)

        self.layer5 = nn.Linear(128, out_channels)
        self.ln5 = nn.LayerNorm(out_channels)

        self.layer6 = nn.Linear(out_channels, out_channels)
        self.ln6 = nn.LayerNorm(out_channels)

    def forward(self, x):
        N, T, S, D = x.shape
        x = rearrange(x, "N T S D -> N S (T D)")
        y = F.relu(self.ln1(self.layer1(x)))
        y = F.relu(self.ln2(self.layer2(y)))
        y = F.relu(self.ln3(self.layer3(y)))

        c = y.amax(dim=1, keepdim=True)
        c = c.repeat(1, S, 1)
        x = torch.cat([x, c], dim=-1)

        y = F.relu(self.ln4(self.layer4(x)))
        y = F.relu(self.ln5(self.layer5(y)))
        y = F.relu(self.ln6(self.layer6(y)))
        return y


def test_ust():
    N, T, S, D = 2, 10, 5, 16
    x = torch.randn(N, T, S, D)
    m = USTEncoder(T*D, 32)
    y = m(x)
    print(y.shape)


if __name__ == "__main__":
    test_ust()
