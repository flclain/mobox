import math
import torch
import torch.nn as nn

from einops import repeat


def get_sine_pos_embed(pos_tensor: torch.Tensor,
                       num_pos_feats: int = 128,
                       temperature: int = 10000,
                       exchange_xy: bool = True) -> torch.Tensor:
    """Generate sine position embedding from a position tensor.

    Reference: https://github.com/IDEA-Research/detrex/blob/main/detrex/layers/position_embedding.py#L178

    Args:
      pos_tensor (tensor): Shape as `(None, n)`.
      num_pos_feats (int): projected shape for each float in the tensor. Default: 128
      temperature (int): The temperature used for scaling the position embedding. Default: 10000.
      exchange_xy (bool, optional): exchange pos x and pos y. 
        For example, input tensor is `[x, y]`, the results will be `[pos(y), pos(x)]`. Defaults: True.

    Returns:
      (tensor) returned position embedding with shape `(None, n * num_pos_feats)`.
    """
    scale = 2 * math.pi
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos_tensor.device)
    dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / num_pos_feats)

    def sine_func(x: torch.Tensor):
        sin_x = x * scale / dim_t
        sin_x = torch.stack((sin_x[:, :, 0::2].sin(), sin_x[:, :, 1::2].cos()), dim=3).flatten(2)
        return sin_x

    pos_res = [sine_func(x) for x in pos_tensor.split([1] * pos_tensor.shape[-1], dim=-1)]
    if exchange_xy:
        pos_res[0], pos_res[1] = pos_res[1], pos_res[0]
    pos_res = torch.cat(pos_res, dim=2)
    return pos_res


def positional_encoding(length, depth):
    """Sinusoidal position encoding.

    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    positions = torch.arange(0, length)[:, None]
    depth = depth // 2
    depths = torch.arange(0, depth)[None, :] / depth

    angle_rates = 1. / (10000.**depths)   # [1,N]
    angle_rads = positions * angle_rates  # [L,D]

    pos_encoding = torch.stack([angle_rads.sin(), angle_rads.cos()], dim=-1)
    return pos_encoding.reshape(length, -1)


class LearnedTimeEncoding(nn.Module):
    def __init__(self, num_pos, d_model):
        super().__init__()
        self.T_embed = nn.Embedding(num_pos, d_model)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.T_embed.weight, 0.0)

    def forward(self, x):
        N, T, S, D = x.shape
        t = torch.arange(T, device=x.device)
        t_emb = self.T_embed(t)  # [T,D]
        y = x + t_emb.unsqueeze(1)
        return y


class LearnedPosEncoding(nn.Module):
    def __init__(self, num_pos, d_model):
        super().__init__()
        self.L_embed = nn.Embedding(num_pos, d_model)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.L_embed.weight, 0.0)

    def forward(self, x):
        N, L, D = x.shape
        t = torch.arange(L, device=x.device)
        pos_emb = self.L_embed(t)  # [L,D]
        return repeat(pos_emb, "L D -> N L D", N=N)
