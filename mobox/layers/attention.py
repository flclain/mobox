import torch
import torch.nn as nn
import torch.nn.functional as F

import warnings
import numpy as np

from typing import Optional
from einops import rearrange


class MultiheadAttention(nn.Module):
    """A wrapper for ``torch.nn.MultiheadAttention``

    Implemente MultiheadAttention with identity connection,
    and position embedding is also passed as input.

    Args:
        embed_dim (int): The embedding dimension for attention.
        num_heads (int): The number of attention heads.
        attn_drop (float): A Dropout layer on attn_output_weights.
            Default: 0.0.
        proj_drop (float): A Dropout layer after `MultiheadAttention`.
            Default: 0.0.
        batch_first (bool): if `True`, then the input and output tensor will be
            provided as `(bs, n, embed_dim)`. Default: False. `(n, bs, embed_dim)`
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        batch_first: bool = False,
        **kwargs,
    ):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.batch_first = batch_first

        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attn_drop,
            batch_first=batch_first,
            **kwargs,
        )

        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        identity: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
        key_pos: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward function for `MultiheadAttention`

        **kwargs allow passing a more general data flow when combining
        with other operations in `transformerlayer`.

        Args:
            query (torch.Tensor): Query embeddings with shape
                `(num_query, bs, embed_dim)` if self.batch_first is False,
                else `(bs, num_query, embed_dim)`
            key (torch.Tensor): Key embeddings with shape
                `(num_key, bs, embed_dim)` if self.batch_first is False,
                else `(bs, num_key, embed_dim)`
            value (torch.Tensor): Value embeddings with the same shape as `key`.
                Same in `torch.nn.MultiheadAttention.forward`. Default: None.
                If None, the `key` will be used.
            identity (torch.Tensor): The tensor, with the same shape as x, will
                be used for identity addition. Default: None.
                If None, `query` will be used.
            query_pos (torch.Tensor): The position embedding for query, with the
                same shape as `query`. Default: None.
            key_pos (torch.Tensor): The position embedding for key. Default: None.
                If None, and `query_pos` has the same shape as `key`, then `query_pos`
                will be used for `key_pos`.
            attn_mask (torch.Tensor): ByteTensor mask with shape `(num_query, num_key)`.
                Same as `torch.nn.MultiheadAttention.forward`. Default: None.
            key_padding_mask (torch.Tensor): ByteTensor with shape `(bs, num_key)` which
                indicates which elements within `key` to be ignored in attention.
                Default: None.
        """
        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        if key_pos is None:
            if query_pos is not None:
                # use query_pos if key_pos is not available
                if query_pos.shape == key.shape:
                    key_pos = query_pos
                else:
                    warnings.warn(
                        f"position encoding of key is" f"missing in {self.__class__.__name__}."
                    )
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos

        out = self.attn(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )[0]

        return identity + self.proj_drop(out)


class ConditionalSelfAttention(nn.Module):
    """Conditional Self-Attention Module used in Conditional-DETR

    `Conditional DETR for Fast Training Convergence.
    <https://arxiv.org/pdf/2108.06152.pdf>`_


    Args:
        embed_dim (int): The embedding dimension for attention.
        num_heads (int): The number of attention heads.
        attn_drop (float): A Dropout layer on attn_output_weights.
            Default: 0.0.
        proj_drop (float): A Dropout layer after `MultiheadAttention`.
            Default: 0.0.
        batch_first (bool): if `True`, then the input and output tensor will be
            provided as `(bs, n, embed_dim)`. Default: False. `(n, bs, embed_dim)`
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        attn_drop=0.0,
        proj_drop=0.0,
        batch_first=False,
        **kwargs,
    ):
        super(ConditionalSelfAttention, self).__init__()
        self.query_content_proj = nn.Linear(embed_dim, embed_dim)
        self.query_pos_proj = nn.Linear(embed_dim, embed_dim)
        self.key_content_proj = nn.Linear(embed_dim, embed_dim)
        self.key_pos_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        head_dim = embed_dim // num_heads
        self.scale = head_dim**-0.5
        self.batch_first = batch_first

    def forward(
        self,
        query,
        key=None,
        value=None,
        identity=None,
        query_pos=None,
        key_pos=None,
        attn_mask=None,
        key_padding_mask=None,
        **kwargs,
    ):
        """Forward function for `ConditionalSelfAttention`

        **kwargs allow passing a more general data flow when combining
        with other operations in `transformerlayer`.

        Args:
            query (torch.Tensor): Query embeddings with shape
                `(num_query, bs, embed_dim)` if self.batch_first is False,
                else `(bs, num_query, embed_dim)`
            key (torch.Tensor): Key embeddings with shape
                `(num_key, bs, embed_dim)` if self.batch_first is False,
                else `(bs, num_key, embed_dim)`
            value (torch.Tensor): Value embeddings with the same shape as `key`.
                Same in `torch.nn.MultiheadAttention.forward`. Default: None.
                If None, the `key` will be used.
            identity (torch.Tensor): The tensor, with the same shape as `query``,
                which will be used for identity addition. Default: None.
                If None, `query` will be used.
            query_pos (torch.Tensor): The position embedding for query, with the
                same shape as `query`. Default: None.
            key_pos (torch.Tensor): The position embedding for key. Default: None.
                If None, and `query_pos` has the same shape as `key`, then `query_pos`
                will be used for `key_pos`.
            attn_mask (torch.Tensor): ByteTensor mask with shape `(num_query, num_key)`.
                Same as `torch.nn.MultiheadAttention.forward`. Default: None.
            key_padding_mask (torch.Tensor): ByteTensor with shape `(bs, num_key)` which
                indicates which elements within `key` to be ignored in attention.
                Default: None.
        """
        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        if key_pos is None:
            if query_pos is not None:
                # use query_pos if key_pos is not available
                if query_pos.shape == key.shape:
                    key_pos = query_pos
                else:
                    warnings.warn(
                        f"position encoding of key is" f"missing in {self.__class__.__name__}."
                    )

        assert (
            query_pos is not None and key_pos is not None
        ), "query_pos and key_pos must be passed into ConditionalAttention Module"

        # transpose (b n c) to (n b c) for attention calculation
        if self.batch_first:
            query = query.transpose(0, 1)  # (n b c)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
            query_pos = query_pos.transpose(0, 1)
            key_pos = key_pos.transpose(0, 1)
            identity = identity.transpose(0, 1)

        # query/key/value content and position embedding projection
        query_content = self.query_content_proj(query)
        query_pos = self.query_pos_proj(query_pos)
        key_content = self.key_content_proj(key)
        key_pos = self.key_pos_proj(key_pos)
        value = self.value_proj(value)

        # attention calculation
        N, B, C = query_content.shape
        q = query_content + query_pos
        k = key_content + key_pos
        v = value

        q = q.reshape(N, B, self.num_heads, C // self.num_heads).permute(
            1, 2, 0, 3
        )  # (B, num_heads, N, head_dim)
        k = k.reshape(N, B, self.num_heads, C // self.num_heads).permute(1, 2, 0, 3)
        v = v.reshape(N, B, self.num_heads, C // self.num_heads).permute(1, 2, 0, 3)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        # add attention mask
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn.masked_fill_(attn_mask, float("-inf"))
            else:
                attn += attn_mask
        if key_padding_mask is not None:
            attn = attn.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.out_proj(out)

        if not self.batch_first:
            out = out.transpose(0, 1)
        return identity + self.proj_drop(out)


class ConditionalCrossAttention(nn.Module):
    """Conditional Cross-Attention Module used in Conditional-DETR

    `Conditional DETR for Fast Training Convergence.
    <https://arxiv.org/pdf/2108.06152.pdf>`_


    Args:
        embed_dim (int): The embedding dimension for attention.
        num_heads (int): The number of attention heads.
        attn_drop (float): A Dropout layer on attn_output_weights.
            Default: 0.0.
        proj_drop (float): A Dropout layer after `MultiheadAttention`.
            Default: 0.0.
        batch_first (bool): if `True`, then the input and output tensor will be
            provided as `(bs, n, embed_dim)`. Default: False. `(n, bs, embed_dim)`
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        attn_drop=0.0,
        proj_drop=0.0,
        batch_first=False,
        **kwargs,
    ):
        super(ConditionalCrossAttention, self).__init__()
        self.query_content_proj = nn.Linear(embed_dim, embed_dim)
        self.query_pos_proj = nn.Linear(embed_dim, embed_dim)
        self.query_pos_sine_proj = nn.Linear(embed_dim, embed_dim)
        self.key_content_proj = nn.Linear(embed_dim, embed_dim)
        self.key_pos_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.num_heads = num_heads
        self.batch_first = batch_first

    def forward(
        self,
        query,
        key=None,
        value=None,
        identity=None,
        query_pos=None,
        key_pos=None,
        query_sine_embed=None,
        is_first_layer=False,
        attn_mask=None,
        key_padding_mask=None,
        **kwargs,
    ):
        """Forward function for `ConditionalCrossAttention`

        **kwargs allow passing a more general data flow when combining
        with other operations in `transformerlayer`.

        Args:
            query (torch.Tensor): Query embeddings with shape
                `(num_query, bs, embed_dim)` if self.batch_first is False,
                else `(bs, num_query, embed_dim)`
            key (torch.Tensor): Key embeddings with shape
                `(num_key, bs, embed_dim)` if self.batch_first is False,
                else `(bs, num_key, embed_dim)`
            value (torch.Tensor): Value embeddings with the same shape as `key`.
                Same in `torch.nn.MultiheadAttention.forward`. Default: None.
                If None, the `key` will be used.
            identity (torch.Tensor): The tensor, with the same shape as x, will
                be used for identity addition. Default: None.
                If None, `query` will be used.
            query_pos (torch.Tensor): The position embedding for query, with the
                same shape as `query`. Default: None.
            key_pos (torch.Tensor): The position embedding for key. Default: None.
                If None, and `query_pos` has the same shape as `key`, then `query_pos`
                will be used for `key_pos`.
            query_sine_embed (torch.Tensor): None
            is_first_layer (bool): None
            attn_mask (torch.Tensor): ByteTensor mask with shape `(num_query, num_key)`.
                Same as `torch.nn.MultiheadAttention.forward`. Default: None.
            key_padding_mask (torch.Tensor): ByteTensor with shape `(bs, num_key)` which
                indicates which elements within `key` to be ignored in attention.
                Default: None.
        """
        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        if key_pos is None:
            if query_pos is not None:
                # use query_pos if key_pos is not available
                if query_pos.shape == key.shape:
                    key_pos = query_pos
                else:
                    warnings.warn(
                        f"position encoding of key is" f"missing in {self.__class__.__name__}."
                    )

        assert (
            query_pos is not None and key_pos is not None
        ), "query_pos and key_pos must be passed into ConditionalAttention Module"

        # transpose (b n c) to (n b c) for attention calculation
        if self.batch_first:
            query = query.transpose(0, 1)  # (n b c)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
            query_pos = query_pos.transpose(0, 1)
            key_pos = key_pos.transpose(0, 1)
            identity = identity.transpose(0, 1)

        # content projection
        query_content = self.query_content_proj(query)
        key_content = self.key_content_proj(key)
        value = self.value_proj(value)

        # shape info
        N, B, C = query_content.shape
        HW, _, _ = key_content.shape

        # position projection
        key_pos = self.key_pos_proj(key_pos)
        if is_first_layer:
            query_pos = self.query_pos_proj(query_pos)
            q = query_content + query_pos
            k = key_content + key_pos
        else:
            q = query_content
            k = key_content
        v = value

        # preprocess
        q = q.view(N, B, self.num_heads, C // self.num_heads)
        query_sine_embed = self.query_pos_sine_proj(query_sine_embed).view(
            N, B, self.num_heads, C // self.num_heads
        )
        q = torch.cat([q, query_sine_embed], dim=3).view(N, B, C * 2)

        k = k.view(HW, B, self.num_heads, C // self.num_heads)  # N, 16, 256
        key_pos = key_pos.view(HW, B, self.num_heads, C // self.num_heads)
        k = torch.cat([k, key_pos], dim=3).view(HW, B, C * 2)

        # attention calculation
        q = q.reshape(N, B, self.num_heads, C * 2 // self.num_heads).permute(
            1, 2, 0, 3
        )  # (B, num_heads, N, head_dim)
        k = k.reshape(HW, B, self.num_heads, C * 2 // self.num_heads).permute(1, 2, 0, 3)
        v = v.reshape(HW, B, self.num_heads, C // self.num_heads).permute(1, 2, 0, 3)

        scale = (C * 2 // self.num_heads) ** -0.5
        q = q * scale
        attn = q @ k.transpose(-2, -1)

        # add attention mask
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn.masked_fill_(attn_mask, float("-inf"))
            else:
                attn += attn_mask
        if key_padding_mask is not None:
            attn = attn.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.out_proj(out)

        if not self.batch_first:
            out = out.transpose(0, 1)

        return identity + self.proj_drop(out)


# =========================================
# My implementations.
# =========================================

class FactorizedAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, kdim=None, vdim=None):
        super().__init__()
        self.num_heads = num_heads
        self.dropout = dropout

        kdim = embed_dim if kdim is None else kdim
        vdim = embed_dim if vdim is None else vdim
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, kdim)
        self.v_proj = nn.Linear(embed_dim, vdim)
        self.out_proj = nn.Linear(vdim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        """
        Args:
          query: query, sized [N, T, Lq, D].
          key: key, sized [N, T, Lk, D].
          value: value, sized [N, T, Lk, D].
          key_padding_mask: key padding mask, sized [N, T, Lk].
          attn_mask: attention mask, sized [N, T, Lq, Lk].
        """
        N, T, Lk = key.shape[:3]
        query = rearrange(self.q_proj(query), "b T l (head k) -> head b T l k",
                          head=self.num_heads)
        key = rearrange(self.k_proj(key), "b T t (head k) -> head b T t k",
                        head=self.num_heads)
        value = rearrange(self.v_proj(value), "b T t (head v) -> head b T t v",
                          head=self.num_heads)
        attn = torch.einsum("hbTlk,hbTtk->hbTlt",
                            [query, key]) / np.sqrt(query.shape[-1])

        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.view(1, N, T, 1, Lk)
            attn_mask = key_padding_mask

        if attn_mask is not None:
            attn.masked_fill_(attn_mask, -np.inf)

        attn = torch.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        if self.dropout > 0:
            attn = F.dropout(attn, p=self.dropout)
        output = torch.einsum("hbTlt,hbTtv->hbTlv", [attn, value])
        output = rearrange(output, "head b T l v -> b T l (head v)")
        output = self.out_proj(output)
        return output, attn


class MyConditionalMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, kdim=None, vdim=None):
        super().__init__()
        self.num_heads = num_heads
        self.dropout = dropout

        kdim = embed_dim if kdim is None else kdim
        vdim = embed_dim if vdim is None else vdim
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, kdim)
        self.q_proj2 = nn.Linear(embed_dim, embed_dim)
        self.k_proj2 = nn.Linear(embed_dim, kdim)
        self.v_proj = nn.Linear(embed_dim, vdim)
        self.out_proj = nn.Linear(vdim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, q1, q2, k1, k2, v, key_padding_mask=None, attn_mask=None):
        """
        Args:
          query: query, sized [N, Lq, 2*D].
          key: key, sized [N, Lk, 2*D].
          value: value, sized [N, Lk, D].
          key_padding_mask: key padding mask, sized [N, Lk].
          attn_mask: attention mask, sized [N, Lq, Lk].
        """
        N, Lk = k1.shape[:2]
        q1 = rearrange(self.q_proj(q1), "b l (head k) -> head b l k",
                       head=self.num_heads)
        k1 = rearrange(self.k_proj(k1), "b t (head k) -> head b t k",
                       head=self.num_heads)
        q2 = rearrange(self.q_proj2(q2), "b l (head k) -> head b l k",
                       head=self.num_heads)
        k2 = rearrange(self.k_proj2(k2), "b t (head k) -> head b t k",
                       head=self.num_heads)
        v = rearrange(self.v_proj(v), "b t (head v) -> head b t v",
                      head=self.num_heads)
        attn1 = torch.einsum(
            "hblk,hbtk->hblt", [q1, k1]) / np.sqrt(q1.shape[-1])
        attn2 = torch.einsum(
            "hblk,hbtk->hblt", [q2, k2]) / np.sqrt(q2.shape[-1])
        attn = attn1 + attn2

        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.view(1, N, 1, Lk)
            attn_mask = key_padding_mask

        if attn_mask is not None:
            attn.masked_fill_(attn_mask, -np.inf)

        attn = torch.softmax(attn, dim=3)
        if self.dropout > 0:
            attn = F.dropout(attn, p=self.dropout)
        output = torch.einsum("hblt,hbtv->hblv", [attn, v])
        output = rearrange(output, "head b l v -> b l (head v)")
        output = self.out_proj(output)
        return output, attn


class MyMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, kdim=None, vdim=None):
        super().__init__()
        self.num_heads = num_heads
        self.dropout = dropout

        kdim = embed_dim if kdim is None else kdim
        vdim = embed_dim if vdim is None else vdim
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, kdim)
        self.v_proj = nn.Linear(embed_dim, vdim)

        self.out_proj = nn.Linear(vdim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        """
        Args:
          query: query, sized [N, Lq, D].
          key: key, sized [N, Lk, D].
          value: value, sized [N, Lk, D].
          key_padding_mask: key padding mask, sized [N, Lk].
          attn_mask: attention mask, sized [N, Lq, Lk].
        """
        N, Lk = key.shape[:2]
        query = rearrange(self.q_proj(query), "b l (head k) -> head b l k",
                          head=self.num_heads)
        key = rearrange(self.k_proj(key), "b t (head k) -> head b t k",
                        head=self.num_heads)
        value = rearrange(self.v_proj(value), "b t (head v) -> head b t v",
                          head=self.num_heads)
        attn = torch.einsum("hblk,hbtk->hblt",
                            [query, key]) / np.sqrt(query.shape[-1])

        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.view(1, N, 1, Lk)
            attn_mask = key_padding_mask

        if attn_mask is not None:
            attn.masked_fill_(attn_mask, -np.inf)

        attn = torch.softmax(attn, dim=3)
        attn = torch.nan_to_num(attn, nan=0.0)
        if self.dropout > 0:
            attn = F.dropout(attn, p=self.dropout)
        output = torch.einsum("hblt,hbtv->hblv", [attn, value])
        output = rearrange(output, "head b l v -> b l (head v)")
        output = self.out_proj(output)
        return output, attn


def test_factorized_attention():
    N = 4
    T = 16
    D = 128
    Lq = 10
    Lk = 8
    nhead = 2
    q = torch.randn(N, T, Lq, D)
    k = torch.randn(N, T, Lk, D)
    v = torch.randn(N, T, Lk, D)

    m = FactorizedAttention(
        embed_dim=D, num_heads=nhead,  kdim=D, vdim=D)
    key_padding_mask = torch.randn(N, T, Lk) > 0
    y, a = m(q, k, v, key_padding_mask=key_padding_mask)
    print(y.shape)
    print(a.shape)


def test_multihead_attention():
    N = 4
    D = 128
    Lq = 10
    Lk = 8
    nhead = 2
    q = torch.randn(N, Lq, D)
    k = torch.randn(N, Lk, D)
    v = torch.randn(N, Lk, D)

    m = MultiHeadAttention(embed_dim=D, num_heads=nhead,  kdim=D, vdim=D)
    # attn_mask = torch.randn(N, Lq, Lk) > 0
    # y, a = m(q, k, v, attn_mask=attn_mask)
    key_padding_mask = torch.randn(N, Lk) > 0
    y, a = m(q, k, v, key_padding_mask=key_padding_mask)
    print(y.shape)
    print(a.shape)


if __name__ == "__main__":
    # test_multihead_attention()
    test_factorized_attention()
