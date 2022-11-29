"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers

Reference: https://github.com/facebookresearch/detr/blob/main/models/transformer.py
"""
import copy
import torch
import torch.nn.functional as F

from einops import repeat
from typing import Optional

from torch import nn, Tensor
from mobox.layers.mlp import MLP
from mobox.layers.position_encoding import get_sine_pos_embed


class Transformer(nn.Module):
    def __init__(self, d_model, num_heads=2, num_layers=2):
        super().__init__()
        encoder_layer = TransformerEncoderLayer(d_model, num_heads, dim_feedforward=1024, dropout=0)
        self.encoder = TransformerEncoder(encoder_layer, num_layers)

        decoder_layer = TransformerDecoderLayer(d_model, num_heads, dim_feedforward=1024, dropout=0)
        self.decoder = TransformerDecoder(decoder_layer, num_layers)

        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, attn_mask, query_embed, pos_embed):
        N = x.size(0)
        memory = self.encoder(x, x, x, query_pos=pos_embed, key_pos=pos_embed, attn_mask=attn_mask)
        query_embed = repeat(query_embed, "L D -> N L D", N=N)
        target = torch.zeros_like(query_embed)
        out = self.decoder(target, memory, memory, query_pos=query_embed, key_pos=pos_embed)
        return out


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, query, key, value,
                query_pos=None, key_pos=None,
                attn_mask=None, key_padding_mask=None):
        """For self-attention encoder: query=key=value, query_pos=key_pos."""
        for layer in self.layers:
            query = layer(query, key, value,
                          query_pos=query_pos, key_pos=key_pos,
                          attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        if self.norm is not None:
            query = self.norm(query)
        return query


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward=2048, dropout=0, activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def add_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, query, key, value,
                query_pos=None, key_pos=None,
                attn_mask=None, key_padding_mask=None):
        q = self.add_pos_embed(query, query_pos)
        k = self.add_pos_embed(key, key_pos)
        out = self.self_attn(q, k, value=value, attn_mask=attn_mask,
                             key_padding_mask=key_padding_mask)[0]
        out = query + self.dropout1(out)
        out = self.norm1(out)

        out2 = self.linear2(self.dropout(self.activation(self.linear1(out))))
        out = out + self.dropout2(out2)
        out = self.norm2(out)
        return out


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, query, key, value,
                query_pos=None, key_pos=None,
                self_attn_mask=None, self_key_padding_mask=None,
                cross_attn_mask=None, cross_key_padding_mask=None):
        for layer in self.layers:
            query = layer(query, key, value,
                          self_query_pos=query_pos,
                          self_attn_mask=self_attn_mask,
                          self_key_padding_mask=self_key_padding_mask,
                          cross_query_pos=query_pos,
                          cross_key_pos=key_pos,
                          cross_attn_mask=cross_attn_mask,
                          cross_key_padding_mask=cross_key_padding_mask)
        if self.norm is not None:
            query = self.norm(query)
        return query


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward=2048, dropout=0, activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def add_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def cat_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else torch.cat([tensor,  pos], dim=-1)

    def forward(self, query, key, value,
                self_query_pos=None,  # self_key_pos = self_query_pos
                self_attn_mask=None,
                self_key_padding_mask=None,
                cross_query_pos=None,
                cross_key_pos=None,
                cross_attn_mask=None,
                cross_key_padding_mask=None):
        q = k = self.add_pos_embed(query, self_query_pos)
        out = self.self_attn(q, k, value=query, attn_mask=self_attn_mask,
                             key_padding_mask=self_key_padding_mask)[0]
        query = query + self.dropout1(out)
        query = self.norm1(query)

        out = self.cross_attn(query=self.add_pos_embed(query, cross_query_pos),
                              key=self.add_pos_embed(key, cross_key_pos),
                              value=value, attn_mask=cross_attn_mask,
                              key_padding_mask=cross_key_padding_mask)[0]
        out = query + self.dropout2(out)
        out = self.norm2(out)

        out2 = self.linear2(self.dropout(self.activation(self.linear1(out))))
        out = out + self.dropout3(out2)
        out = self.norm3(out)
        return out


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
