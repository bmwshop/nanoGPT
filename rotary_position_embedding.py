import torch
from einops import rearrange
from torch import einsum, nn

__all__ = ['RotaryEmbedding', 'apply_rotary_pos_emb']

def _rotate_half(x):
    """
    change sign so the last dimension
    [A, B, C, D] -> [-C, -D, A, B]
    """
    x = rearrange(x, '... (j d) -> ... j d', j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)

class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        dim: int,
        rotary_base: int = 10000,
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

    def forward(self, max_seq_len):
        seq = torch.arange(max_seq_len, device=self.inv_freq.device)
        seq = seq.type_as(self.inv_freq)

        freqs = einsum('i , j -> i j', seq, self.inv_freq)
        # first part even vector components, second part odd vector components,
        #  2 * dim in dimension size
        emb = torch.cat((freqs, freqs), dim=-1)
        # emb [seq_length, .., dim]
        # return rearrange(emb, 'n d -> n 1 1 d')
        return emb

def apply_rotary_pos_emb(t, freqs):
    # def apply(self, t):
        """
        input tensor t is of shape [seq_length, ..., dim]
        rotary positional embeding tensor freqs is of shape [seq_length, ..., dim]
        check https://kexue.fm/archives/8265 for detailed formulas
        """
        rot_dim = freqs.shape[-1]
        # ideally t_pass is empty so rotary pos embedding is applied to all tensor t
        t, t_pass = t[..., :rot_dim], t[..., rot_dim:]
        # first part is cosine component
        # second part is sine component, need to change signs with _rotate_half method
        t = (t * freqs.cos()) + (_rotate_half(t) * freqs.sin())
        return torch.cat((t, t_pass), dim=-1)

