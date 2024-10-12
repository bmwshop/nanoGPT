import torch
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import logging

def _rotate_gptj(x: torch.Tensor) -> torch.Tensor:
  # Dima: the only effect of the below appears to be to
  # negate the odd-indexed elements in the last dimension.
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)

class SimpleRotaryEmbedding():

    """Simple rotary positional embedding."""

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.dtype = dtype

        self.cos_sin_cache = self._compute_cos_sin_cache()
        self.cos_sin_cache = self.cos_sin_cache.to(dtype)


    def _compute_cos_sin_cache(self) -> torch.Tensor:

        # Dima: forcing float here because of precision issues
        inv_freq = 1.0 / (self.base**(torch.arange(
            0, self.rotary_dim, 2, dtype=torch.float) / self.rotary_dim))

        t = torch.arange(self.max_position_embeddings, dtype=torch.float)

        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)
        return cache

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        query = query.view(*query.shape[:-1], -1, self.head_size)
        key = key.view(*key.shape[:-1], -1, self.head_size)

        query_rot = query[..., :self.rotary_dim]
        key_rot = key[..., :self.rotary_dim]
        if self.rotary_dim < self.head_size:
            query_pass = query[..., self.rotary_dim:]
            key_pass = key[..., self.rotary_dim:]

        self.cos_sin_cache: torch.Tensor = self.cos_sin_cache.to(
            positions.device, dtype=query.dtype)

        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)

        cos = cos.repeat_interleave(2, dim=-1).unsqueeze(-2)
        sin = sin.repeat_interleave(2, dim=-1).unsqueeze(-2)

        query_rot = query_rot * cos + _rotate_gptj(query_rot) * sin
        key_rot = key_rot * cos + _rotate_gptj(key_rot) * sin

        if self.rotary_dim < self.head_size:
            query = torch.cat((query_rot, query_pass), dim=-1)
            key = torch.cat((key_rot, key_pass), dim=-1)
        else:
            query = query_rot
            key = key_rot
        query = query.flatten(-2)
        key = key.flatten(-2)
        return query, key
