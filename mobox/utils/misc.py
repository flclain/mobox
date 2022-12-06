import torch


class NestedTensor:
    tensor: torch.Tensor
    mask: torch.Tensor
    device: torch.device

    def __init__(self, tensor, mask):
        self.tensor = tensor
        self.mask = mask
        self.device = tensor.device

    def to(self, device):
        self.tensor = self.tensor.to(device)
        self.mask = self.mask.to(device)
        self.device = device
        return self

    def cuda(self):
        return self.to("cuda:0")

    def unbind(self):
        return self.tensor, self.mask

    @property
    def shape(self):
        return self.tensor.shape

    @staticmethod
    def from_numpy(tensor, mask):
        return NestedTensor(torch.from_numpy(tensor), torch.from_numpy(mask))

    def __str__(self):
        return f"NestedTensor[tensor: {self.tensor.shape}, mask: {self.mask.shape}]"

    def __repr__(self):
        return self.__str__()


if __name__ == "__main__":
    import numpy as np
    x = torch.randn(2, 3)
    m = x > 0
    nt = NestedTensor(x, m)
    print(nt.tensor)
    print(nt.mask)
    print(nt.device)

    x2 = np.random.randn(2, 3)
    m2 = x2 > 0
    nt2 = NestedTensor.from_numpy(x2, m2)
    print(nt2.tensor)
    print(nt2.mask)
    print(nt.device)
