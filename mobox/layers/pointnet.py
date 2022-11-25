import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange


class STN(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.conv1 = nn.Conv1d(d_model, 64, kernel_size=1,
                               stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1,
                               stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 1024, kernel_size=1,
                               stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm1d(1024)
        self.fc1 = nn.Linear(1024, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn5 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, d_model*d_model)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.max(dim=-1)[0]
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        x = x.reshape(-1, self.d_model, self.d_model)
        y = x + torch.eye(self.d_model, device=x.device)
        return y


class PointNetEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.stn1 = STN(in_channels)
        self.conv1 = nn.Conv1d(
            in_channels, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm1d(64)

        self.stn2 = STN(64)
        self.conv2 = nn.Conv1d(64, 1024, kernel_size=1,
                               stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm1d(1024)

        self.conv3 = nn.Conv1d(
            1024, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        N, T, S, D = x.shape
        x = rearrange(x, "n t s d -> (n s) d t")
        t = self.stn1(x)
        x = torch.einsum("ndt,ndl->nlt", [x, t])
        x = F.relu(self.bn1(self.conv1(x)))
        t = self.stn2(x)
        x = torch.einsum("ndt,ndl->nlt", [x, t])
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = x.max(dim=-1)[0]  # [NS, D]
        y = rearrange(x, "(n s) d -> n s d", n=N)
        return y


class PolylineEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm1d(256)

        self.conv2 = nn.Conv1d(256, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm1d(256)

        self.conv3 = nn.Conv1d(256, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm1d(out_channels)

        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        N, T, S, D = x.shape
        x = rearrange(x, "n t s d -> (n s) d t")
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.max(dim=-1)[0]  # [NS, D]
        y = rearrange(x, "(n s) d -> n s d", n=N)
        return y


def test_pointnet():
    N = 8
    T = 10
    S = 128
    D = 32
    m = PolylineEncoder(D, 256)
    x = torch.randn(N, T, S, D)
    print(x.shape)
    y = m(x)
    print(y.shape)


if __name__ == "__main__":
    test_pointnet()
