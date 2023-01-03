"""VectorNet Polyline Encoder.

Reference: VectorNet: Encoding HD Maps and Agent Dynamics from Vectorized Representation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layer = nn.Linear(in_channels, out_channels)
        self.ln = nn.LayerNorm(out_channels)

    def forward(self, x):
        N, T, S, D = x.shape
        x = F.relu(self.ln(self.layer(x)))  # [N,T,S,D]
        c = x.amax(dim=1, keepdim=True)     # [N,1,S,D]
        c = c.repeat(1, T, 1, 1)
        x = torch.cat([x, c], dim=-1)       # [N,T,S,2*D]
        return x


class VectorNetEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        out_channels = out_channels // 2
        self.layer1 = EncoderLayer(in_channels, 64)
        self.layer2 = EncoderLayer(64*2, 64)
        self.layer3 = EncoderLayer(64*2, 128)
        self.layer4 = EncoderLayer(128*2, 128)
        self.layer5 = EncoderLayer(128*2, out_channels)
        self.layer6 = EncoderLayer(out_channels*2, out_channels)

    def forward(self, x):
        x = self.layer1(x)  # [N,T,S,D]
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = x.amax(dim=1)   # [N,S,D]
        return x


def test_ust():
    N, T, S, D = 2, 10, 5, 16
    x = torch.randn(N, T, S, D)
    m = VectorNetEncoder(D, 128)
    y = m(x)
    print(y.shape)


if __name__ == "__main__":
    test_ust()
