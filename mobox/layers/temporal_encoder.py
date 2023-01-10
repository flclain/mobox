"""Temporal Encoder borrowed from HiVT.

Reference:
  https://github.com/ZikangZhou/HiVT/blob/main/models/local_encoder.py#L218
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional
from einops import rearrange
from mobox.utils.misc import NestedTensor


class TemporalEncoderLayer(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.linear1 = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(embed_dim, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self,
                src: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = src
        x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
        x = x + self._ff_block(self.norm2(x))
        return x

    def _sa_block(self,
                  x: torch.Tensor,
                  attn_mask: Optional[torch.Tensor],
                  key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        x = self.self_attn(x, x, x, attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask, need_weights=False)[0]
        return self.dropout1(x)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(self.dropout(F.relu_(self.linear1(x))))
        return self.dropout2(x)


class TemporalEncoder(nn.Module):
    def __init__(self,
                 input_dim: int,
                 embed_dim: int,
                 historical_steps: int,
                 num_heads: int = 4,
                 num_layers: int = 2,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.proj_layer = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(True),
        )

        encoder_layer = TemporalEncoderLayer(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers,
                                                         norm=nn.LayerNorm(embed_dim))
        self.padding_token = nn.Parameter(torch.Tensor(historical_steps, 1, embed_dim))
        self.cls_token = nn.Parameter(torch.Tensor(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.Tensor(historical_steps + 1, 1, embed_dim))
        attn_mask = self.generate_square_subsequent_mask(historical_steps + 1)
        self.register_buffer("attn_mask", attn_mask)
        nn.init.normal_(self.padding_token, mean=0, std=0.02)
        nn.init.normal_(self.cls_token, mean=0, std=0.02)
        nn.init.normal_(self.pos_embed, mean=0, std=0.02)

    def forward(self, x: NestedTensor) -> torch.Tensor:
        x, mask = x.unbind()
        x = self.proj_layer(x)            # [N,T,S,D]
        padding_mask = (1 - mask).bool()  # [N,T,S]

        N, T, S, D = x.shape
        x = rearrange(x, "N T S D -> T (N S) D")
        padding_mask = rearrange(padding_mask, "N T S -> T (N S) 1")

        x = torch.where(padding_mask, self.padding_token, x)
        expand_cls_token = self.cls_token.expand(-1, x.shape[1], -1)
        x = torch.cat((x, expand_cls_token), dim=0)
        x = x + self.pos_embed
        out = self.transformer_encoder(src=x, mask=self.attn_mask, src_key_padding_mask=None)
        ret = out[-1].reshape(N, S, -1)
        return ret

    @staticmethod
    def generate_square_subsequent_mask(seq_len: int) -> torch.Tensor:
        mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
        return mask
