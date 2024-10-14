import torch
from einops import rearrange
from torch import einsum, nn
import math
import logging

__all__ = ['Xpos2Embedding', 'apply_xpos2_emb']

def _rotate_half(x):
    """
    change sign so the last dimension
    [A, B, C, D] -> [-C, -D, A, B]
    """
    x = rearrange(x, '... (j d) -> ... j d', j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)

class Xpos2Embedding(nn.Module):

    def __init__(
        self,
        dim: int,
        rotary_base: int = 10000,
        max_pos: int = 4096,
        decay_base: int = 2,
        decay_angle: float = torch.pi / 2,
        precision: str = 'bfloat16',
        adaptive = False,
    ):
        """
        Args:

            dim (int): rotary embedding dimension
            rotary_base (int): rotary_base for the positional frequency (default: 10000)
        """
        super().__init__()
        self.rotary_base = rotary_base
        inv_freq = 1.0 / (self.rotary_base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        self.decay_base = decay_base
        self.ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[precision]
        
        if adaptive:
            
            finfo = torch.finfo(ptdtype)
            decay_angles = max_pos * self.inv_freq / math.log(finfo.max, self.decay_base)
            decay_angles = torch.clamp(decay_angles, min = decay_angle)
            logging.debug(f'adaptive decay angles: {decay_angles}')
        else:
             decay_angles = decay_angle
        decay_angles = decay_angles.to(self.ptdtype)
        self.register_buffer('decay_angles', decay_angles)

    def forward(self, max_seq_len):
        seq = torch.arange(max_seq_len, device=self.inv_freq.device)
        # seq = seq.type_as(self.inv_freq)

        angles = einsum('i , j -> i j', seq, self.inv_freq)
        # first part even vector components, second part odd vector components,
        #  2 * dim in dimension size
        logging.debug(f'adaptive decay angles: {self.decay_angles}')
        logging.debug(f'angles: {angles}')
        logging.debug(f'self.decay_base: {self.decay_base}')
        scales = self.decay_base**(-angles/self.decay_angles).to(self.ptdtype)
        logging.debug(f'scales: {scales}')
        emb = torch.cat((angles, angles), dim=-1)
        emb1 = torch.cat((scales, scales), dim=-1)
        # emb [seq_length, .., dim]
        # return rearrange(emb, 'n d -> n 1 1 d')
        
        return emb,emb1

def apply_xpos2_emb(q,k, angles, scales):
        rot_dim = angles.shape[-1]

        q_rot, q_pass = q[..., :rot_dim], q[..., rot_dim:]
        k_rot, k_pass = k[..., :rot_dim], k[..., rot_dim:]


        q_rot = (q_rot * angles.cos() * scales) + (_rotate_half(q_rot) * angles.sin() * scales)
        k_rot = (k_rot * angles.cos() * 1/scales) + (_rotate_half(k_rot) * angles.sin() * 1/scales)

        q = torch.cat((q_rot, q_pass), dim=-1)
        k = torch.cat((k_rot, k_pass), dim=-1)
        
        return q,k

