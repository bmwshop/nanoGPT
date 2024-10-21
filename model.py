"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

from rotary_position_embedding import RotaryEmbedding, apply_rotary_pos_emb
from xpos2_position_embedding import Xpos2Embedding, apply_xpos2_emb
from alibi_relative_position_embedding import build_slopes, build_relative_position
try:
    from flash_attn import flash_attn_func
except ImportError:
    flash_attn_func = None
import logging

from tqdm import tqdm
import heapq

class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # Key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.config = config

        self.alibi_slopes = None
        head_size = self.n_embd // self.n_head

        if config.pe == 'rope':
            logging.debug(f'Initializing RoPE with base {config.rope_base}')
            self.rotary_pos_emb = RotaryEmbedding(head_size, rotary_base=config.rope_base)
        elif config.pe == 'xpos2':
            max_xpos2_pos = config.block_size * 10  # Some buffer
            self.rotary_pos_emb = Xpos2Embedding(
                head_size, rotary_base=config.rope_base,
                max_pos=max_xpos2_pos, decay_base=config.xpos2_decay_base,
                decay_angle=config.xpos2_decay_angle,
                precision=config.precision, adaptive=config.xpos2_adaptive
            )
        elif config.pe == 'alibi':
            self.alibi_slopes = build_slopes(
                num_attention_heads=config.n_head,
                num_attention_heads_alibi=config.n_head,  # It is a useful option to have not to rotate all alibi dims
            ).float()  # Shape: (nheads, 1, 1)
            if config.flash:
                self.alibi_slopes = self.alibi_slopes.squeeze() # reverse the double unsqueeze when creating slopes

        if config.flash:
            # Flash attention makes GPU go brrrr but support is only in PyTorch >= 2.0
            self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        else:
            logging.info('Flash is turned off. GPUs will not go brrrr')
            self.flash = False

        if not self.flash:
            logging.warning("Using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            ##  self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
            ##                              .view(1, 1, config.block_size, config.block_size))
            # D.R. making it just a parameter so that FA checkpoints are compatible with non-FA checkpoints
            self.bias = torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size)

    def forward(self, x, collect_info=False):
        B, T, C = x.size()  # Batch size, sequence length, embedding dimensionality (n_embd)

        # Calculate query, key, values for all heads and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # Shape: (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # Shape: (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # Shape: (B, nh, T, hs)

        if self.config.pe == 'rope':
            # This call expects shape [seq_length, ..., dim]
            angles = self.rotary_pos_emb(q.shape[-2])  # Shape: (T, hs)
            q = apply_rotary_pos_emb(q, angles)
            k = apply_rotary_pos_emb(k, angles)
        elif self.config.pe == 'xpos2':
            # This call expects shape [seq_length, ..., dim]
            angles, scales = self.rotary_pos_emb(q.shape[-2])  # Shape: (T, hs)
            q, k = apply_xpos2_emb(q, k, angles, scales)

        # Scaling the query vectors based on position indices
        if self.training and self.config.scaling_target_sequence_length is not None:
            a = float(self.config.block_size)  # Training sequence length
            b = float(self.config.scaling_target_sequence_length)  # Target sequence length
            T = q.size(2)  # Current sequence length
            if T > 1:
                i = torch.arange(T, device=q.device, dtype=q.dtype).unsqueeze(0)  # Shape: (1, T)
                scaling_factor = 1 + ((a / b) - 1) * (i / (T - 1))  # Shape: (1, T)
                scaling_factor = scaling_factor.view(1, 1, T, 1)  # Reshape for broadcasting to (1, 1, T, 1)
            else:
                scaling_factor = torch.tensor(1.0, device=q.device, dtype=q.dtype).view(1, 1, 1, 1)
            q = q * scaling_factor
        elif self.config.softmax_log_k > 0:  # D.R. log-based scaling formula
            if T > 1:
                _k = float(self.config.softmax_log_k)
                i = torch.arange(T, device=q.device, dtype=q.dtype).unsqueeze(0)  # Shape: (1, T)
                i[0] = 1 # avoid -inf
                scaling_factor = (1 - _k + _k * torch.log(i)).to(q.dtype)   
                scaling_factor = scaling_factor.view(1, 1, T, 1)  # Reshape for broadcasting to (1, 1, T, 1)
            else:
                scaling_factor = torch.tensor(1.0, device=q.device, dtype=q.dtype).view(1, 1, 1, 1)
            q = q * scaling_factor

        # Collect norms
        q_norms = q.norm(dim=-1)  # Shape: (B, nh, T)
        k_norms = k.norm(dim=-1)  # Shape: (B, nh, T)
        v_norms = v.norm(dim=-1)  # Shape: (B, nh, T)
        embedding_norms = x.norm(dim=-1).unsqueeze(1).expand(-1, self.n_head, -1)  # Shape: (B, nh, T)

        # Causal self-attention
        if self.flash:
            y = flash_attn_func(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                dropout_p=self.dropout if self.training else 0, softmax_scale=None, causal=True,
                                window_size=(-1, -1), alibi_slopes=self.alibi_slopes, deterministic=False)
            weighted_v = y.transpose(1, 2)
        else:
            # Manual implementation of attention
            att_scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # Shape: (B, nh, T, T)

            if self.config.pe == 'alibi':
                # Implement ALiBi positional bias
                position_matrix = build_relative_position(T, full=True).unsqueeze(0).expand(self.n_head, -1, -1) # nh, T, T
                # alibi slopes shape: (nheads, 1, 1)
                alibi_bias = self.alibi_slopes * position_matrix # nh, T, T
                alibi_bias = alibi_bias.unsqueeze(0).expand(B, -1, -1, -1)  # Shape: (B, nh, T, 1)
                att_scores = att_scores - alibi_bias

            if self.bias.device != att_scores.device:
                self.bias = self.bias.to(att_scores.device)
            att_scores = att_scores.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))

            att_probs = F.softmax(att_scores, dim=-1)  # Shape: (B, nh, T, T)
            att_probs = self.attn_dropout(att_probs)

            if collect_info:
                weighted_v = att_probs @ v  # Shape: (B, nh, T, hs)
                weighted_v_norms = weighted_v.norm(dim=-1)  # Shape: (B, nh, T)

                # Compute weighted_v_norms excluding top 5 attended tokens
                effective_top_k = min(5, T)
                topk_values, topk_indices = torch.topk(att_probs, k=effective_top_k, dim=-1)
                att_probs_excl_topk = att_probs.clone()

                # Zero out topk attention probabilities
                att_probs_excl_topk.scatter_(
                    dim=-1,
                    index=topk_indices,
                    value=0.0
                )

                # Do NOT renormalize the attention probabilities
                # This ensures the sum of attention probabilities is less than or equal to 1

                # Compute weighted_v excluding topk and its norm
                weighted_v_excl_topk = att_probs_excl_topk @ v  # Shape: (B, nh, T, hs)
                weighted_v_excl_topk_norms = weighted_v_excl_topk.norm(dim=-1)  # Shape: (B, nh, T)
            else:
                weighted_v = att_probs @ v  # Shape: (B, nh, T, hs)
                weighted_v_norms = None
                weighted_v_excl_topk_norms = None

        y = weighted_v.transpose(1, 2).contiguous().view(B, T, C)  # Re-assemble all head outputs side by side

        # Output projection
        y = self.resid_dropout(self.c_proj(y))

        extra_info = None
        if collect_info:
            # Adjust top_k based on available tokens
            effective_top_k = min(5, T)

            # Get top_k attention probabilities and indices
            topk_values_to, topk_indices_to = torch.topk(att_probs, k=effective_top_k, dim=-1)  # Shape: (B, nh, T, k)

            # Transpose attention probabilities and apply causal mask
            att_probs_T = att_probs.transpose(-2, -1)  # Shape: (B, nh, T, T)
            causal_mask_T = self.bias[:, :, :T, :T].transpose(-2, -1)
            att_probs_T = att_probs_T * causal_mask_T

            topk_values_from, topk_indices_from = torch.topk(att_probs_T, k=effective_top_k, dim=-1)  # Shape: (B, nh, T, k)

            extra_info = {
                'q_norms': q_norms,  # Shape: (B, nh, T)
                'k_norms': k_norms,  # Shape: (B, nh, T)
                'v_norms': v_norms,  # Shape: (B, nh, T)
                'embedding_norms': embedding_norms,  # Shape: (B, nh, T)
                'weighted_v_norms': weighted_v_norms,  # Shape: (B, nh, T)
                'weighted_v_excl_topk_norms': weighted_v_excl_topk_norms,  # Shape: (B, nh, T)
                'topk_indices_to': topk_indices_to,  # Shape: (B, nh, T, k)
                'topk_values_to': topk_values_to,    # Shape: (B, nh, T, k)
                'topk_indices_from': topk_indices_from,  # Shape: (B, nh, T, k)
                'topk_values_from': topk_values_from,    # Shape: (B, nh, T, k)
            }

        if collect_info:
            return y, extra_info
        else:
            return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, collect_info=False):
        if collect_info:
            attn_out, attn_info = self.attn(self.ln_1(x), collect_info=collect_info)
            x = x + attn_out
            x = x + self.mlp(self.ln_2(x))
            return x, attn_info
        else:
            x = x + self.attn(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
            return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    pe: str = 'abs'  # positional embeddings: 'abs', 'rope', 'alibi', 'nope', 'xpos2'
    flash: bool = False  # Should we use Flash Attention if available?
    rope_base: int = 10000  # RoPE base
    xpos2_decay_base: float = 2.0  # Decay base
    xpos2_decay_angle: float = math.pi / 2  # Soft max angle
    xpos2_adaptive: bool = True  # Should we change decay angle if there's risk of overflow
    precision: str = 'bfloat16'  # Precision
    scaling_target_sequence_length: int = None  # Target sequence length for scaling during training
    softmax_log_k: float = 0.0 # 1/T = =(1−k)⋅1+k⋅log(x) where T = pre-softmax temp

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict()
        self.transformer['wte'] = nn.Embedding(config.vocab_size, config.n_embd)
        self.transformer['drop'] = nn.Dropout(config.dropout)
        self.transformer['h'] = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.transformer['ln_f'] = LayerNorm(config.n_embd, bias=config.bias)

        assert self.config.pe in {'abs', 'rope', 'alibi', 'nope', 'xpos2'}, f"Invalid value for pe: {self.config.pe}"

        if self.config.pe == 'abs':
            logging.info('Using absolute positional embeddings (wpe)')
            self.transformer['wpe'] = nn.Embedding(config.block_size, config.n_embd)
        elif self.config.pe == 'rope':
            logging.info('Using RoPE positional embeddings')
        elif self.config.pe == 'xpos2':
            logging.info('Using XPOS2 positional embeddings')
        elif self.config.pe == 'alibi':
            logging.info('Using ALiBi positional embeddings')
        else:
            logging.info('No positional embeddings used (NoPE)')

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Weight tying
        self.transformer.wte.weight = self.lm_head.weight  # https://paperswithcode.com/method/weight-tying

        # Initialize all weights
        self.apply(self._init_weights)
        # Apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        # Report number of parameters
        logging.info("Number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if self.config.pe == 'abs' and non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None and hasattr(module.bias, 'data'):
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, collect_info=False, collect_probs_per_layer=False):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # Shape: (t)

        # Forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # Shape: (b, t, n_embd)
        if self.config.pe == 'abs':
            pos_emb = self.transformer.wpe(pos)  # Shape: (t, n_embd)
            x = self.transformer.drop(tok_emb + pos_emb)  # Shape: (b, t, n_embd)
        else:
            x = self.transformer.drop(tok_emb)  # Shape: (b, t, n_embd)

        attn_info_per_layer = [] if collect_info else None
        logits_per_layer = [] if collect_probs_per_layer else None  # Initialize list to store logits

        for layer_idx, block in enumerate(self.transformer.h):
            if collect_info:
                x, attn_info = block(x, collect_info=collect_info)
                attn_info_per_layer.append(attn_info)
            else:
                x = block(x)

            if collect_probs_per_layer:
                # Compute logits after this layer
                x0 = self.transformer.ln_f(x)
                if targets is not None:
                    layer_logits = self.lm_head(x0)  # Shape: (b, t, vocab_size)
                else:
                    layer_logits = self.lm_head(x0[:, [-1], :])  # Shape: (b, 1, vocab_size)
                logits_per_layer.append(layer_logits)

        x = self.transformer.ln_f(x)  # Shape: (b, t, n_embd)

        """if collect_probs_per_layer:
            # Compute final logits
            if targets is not None:
                final_logits = self.lm_head(x)  # Shape: (b, t, vocab_size)
            else:
                final_logits = self.lm_head(x[:, [-1], :])  # Shape: (b, 1, vocab_size)
            logits_per_layer.append(final_logits)"""

        if targets is not None:
            # If we are given some desired targets also calculate the loss
            logits = self.lm_head(x)  # Shape: (b, t, vocab_size)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # Inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :])  # Shape: (b, 1, vocab_size)
            loss = None

        if collect_info and collect_probs_per_layer:
            return logits, loss, attn_info_per_layer, logits_per_layer, x
        elif collect_info:
            return logits, loss, attn_info_per_layer, x
        elif collect_probs_per_layer:
            return logits, loss, logits_per_layer
        else:
            return logits, loss

    def crop_block_size(self, block_size):
        # Model surgery to decrease the block size if necessary
        # e.g., we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        if self.config.pe == 'abs':
            self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        logging.info("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        logging.info("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            logging.info(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        logging.info(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        logging.info(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        logging.info(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, collect_info=False,
                 collect_probs_per_layer=False, decode=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        """
        generated_info = [] if collect_info else None
        logits_per_layer_generated = [] if collect_probs_per_layer else None  # Initialize list to store generated logits

        device = idx.device
        batch_size = idx.size(0)
        assert batch_size == 1, "This generate function currently supports batch_size=1 only."

        total_seq_length = idx.size(1) + max_new_tokens
        num_layers = self.config.n_layer
        num_heads = self.config.n_head  # Exclude aggregated head from per-head calculations

        # Initialize attention_scores for each token
        attention_scores = [
            {
                'top_tokens_attending_to': [
                    [{} for _ in range(num_heads)]  # For each layer, list of dicts for each head
                    for _ in range(num_layers)
                ]
            }
            for _ in range(total_seq_length)
        ]

        # Initialize token norms
        token_norms = []

        idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
        seq_len = idx_cond.size(1)
        initial_context_length = seq_len  # The length of the initial context

        # Collect info for initial context
        if collect_info or collect_probs_per_layer:
            outputs = self(idx_cond, collect_info=collect_info, collect_probs_per_layer=collect_probs_per_layer)
            if collect_info and collect_probs_per_layer:
                logits, _, attn_info_per_layer, logits_per_layer, hidden_states = outputs
            elif collect_info:
                logits, _, attn_info_per_layer, hidden_states = outputs
            elif collect_probs_per_layer:
                logits, _, logits_per_layer = outputs
        else:
            logits, _ = self(idx_cond)

        if collect_probs_per_layer:
            logits_per_layer_generated.extend(logits_per_layer)

        if collect_info:
            # Decode tokens individually
            token_ids = idx_cond[0].tolist()  # List of token IDs in the initial context
            decoded_tokens = []
            for token_id in token_ids:
                decoded_token = decode([token_id]) if decode else None
                decoded_tokens.append(decoded_token)

            # Normalize embeddings
            # hidden_states shape: (1, t, n_embd)
            """hidden_states_norm = hidden_states[0] / hidden_states[0].norm(dim=-1, keepdim=True)  # Shape: (t, n_embd)
            embedding_weight_norm = self.lm_head.weight / self.lm_head.weight.norm(dim=-1,
                                                                                   keepdim=True)  # Shape: (vocab_size, n_embd)
            # Compute cosine similarities
            cos_similarities = torch.matmul(hidden_states_norm, embedding_weight_norm.T)  # Shape: (t, vocab_size)
            # For each token, get top 10 most similar tokens
            top_k_similar = 10
            topk_sim_values, topk_sim_indices = torch.topk(cos_similarities, k=top_k_similar, dim=-1)  # Shape: (t, k)"""

            # Initialize token_norms
            token_norms = [{'q_norms': [], 'k_norms': [], 'v_norms': []} for _ in range(seq_len)]

            # Collect norms and attention info
            for layer_idx, layer_attn_info in enumerate(attn_info_per_layer):
                # Collect norms
                q_norms = layer_attn_info['q_norms'][0]  # Shape: (nh, T)
                k_norms = layer_attn_info['k_norms'][0]
                v_norms = layer_attn_info['v_norms'][0]

                for i in range(seq_len):
                    token_norms[i]['q_norms'].append(q_norms[:, i])  # Shape: (nh,)
                    token_norms[i]['k_norms'].append(k_norms[:, i])
                    token_norms[i]['v_norms'].append(v_norms[:, i])

            # Collect token info
            for i in tqdm(range(seq_len)):
                token_id = token_ids[i]
                decoded_token = decoded_tokens[i]
                decoded_token = decoded_token if decoded_token else None

                token_info = {
                    'token_id': token_id,
                    'decoded_token': decoded_token,
                    'is_initial_context': True,
                    'attn_info_per_layer': [],
                    'most_similar_tokens': [],
                    'next_token_probs_per_layer': [],  # Initialize per-layer next token probabilities
                }
                """# Get most similar tokens
                sim_token_ids = topk_sim_indices[i].tolist()
                sim_token_sims = topk_sim_values[i].tolist()
                # Decode similar tokens individually
                sim_decoded_tokens = []
                for sim_tid in sim_token_ids:
                    sim_decoded_token = decode([sim_tid]) if decode else None
                    sim_decoded_tokens.append(sim_decoded_token)"""

                #most_similar_tokens = [
                #    {'token_id': tid, 'decoded_token': dtok, 'similarity': sim}
                #    for tid, dtok, sim in zip(sim_token_ids, sim_decoded_tokens, sim_token_sims)
                #]
                #token_info['most_similar_tokens'] = most_similar_tokens

                for layer_idx, layer_attn_info in enumerate(attn_info_per_layer):
                    layer_token_info = {}
                    for key in [
                        'q_norms', 'k_norms', 'v_norms', 'embedding_norms',
                        'weighted_v_norms', 'weighted_v_excl_topk_norms',
                        'topk_indices_to', 'topk_values_to', 'topk_indices_from', 'topk_values_from'
                    ]:
                        tensor = layer_attn_info[key]
                        if tensor is not None:
                            layer_token_info[key] = tensor[0, :, i]  # Shape depends on key
                        else:
                            layer_token_info[key] = None
                    token_info['attn_info_per_layer'].append(layer_token_info)

                    # Initialize top_tokens_attending_to for tokens in the initial context
                    # Collect top K future tokens within initial context that attend to this token
                    topk_indices_from = layer_attn_info['topk_indices_from'][0, :, i]  # Shape: (nh, k)
                    topk_values_from = layer_attn_info['topk_values_from'][0, :, i]  # Shape: (nh, k)
                    for head_idx in range(num_heads):
                        indices = topk_indices_from[head_idx].tolist()
                        values = topk_values_from[head_idx].tolist()
                        top_tokens_dict = {}
                        for idx_from, attn_score in zip(indices, values):
                            idx_from = int(idx_from)
                            if idx_from <= i or idx_from >= seq_len:
                                continue  # Only consider future tokens within initial context
                            if idx_from in top_tokens_dict:
                                top_tokens_dict[idx_from] += attn_score
                            else:
                                top_tokens_dict[idx_from] = attn_score
                        # Keep top K tokens
                        top_k_attn = 5
                        top_k_tokens = heapq.nlargest(top_k_attn, top_tokens_dict.items(), key=lambda x: x[1])
                        attention_scores[i]['top_tokens_attending_to'][layer_idx][head_idx] = top_k_tokens

                generated_info.append(token_info)

        if collect_info and collect_probs_per_layer:
            # Compute 'next_token_probs_per_layer' for the last token of initial context
            last_token_idx = seq_len - 1
            token_info = generated_info[last_token_idx]  # Get the token_info for the last token
            # Collect next token probabilities per layer
            next_token_probs_per_layer = []
            for layer_logits in logits_per_layer:
                #print(layer_logits.shape)
                layer_logits = layer_logits[:, -1, :]  # Shape: (1, vocab_size)
                layer_probs = F.softmax(layer_logits, dim=-1)
                # Extract top 10 next token probabilities
                top_probs, top_indices = torch.topk(layer_probs, k=10, dim=-1)  # Shape: (1, k)
                next_token_probs_layer = []
                for idx_token, prob in zip(top_indices[0], top_probs[0]):
                    token_id = int(idx_token.item())
                    probability = float(prob.item())
                    decoded_token = decode([token_id]) if decode else None
                    next_token_probs_layer.append({
                        'token_id': token_id,
                        'decoded_token': decoded_token,
                        'probability': probability
                    })
                next_token_probs_per_layer.append(next_token_probs_layer)
            token_info['next_token_probs_per_layer'] = next_token_probs_per_layer

        # Start generating new tokens
        for t in tqdm(range(max_new_tokens), desc="Generating tokens"):
            idx_cond = idx[:, -self.config.block_size:] if idx.size(1) > self.config.block_size else idx
            if collect_info or collect_probs_per_layer:
                outputs = self(idx_cond, collect_info=collect_info, collect_probs_per_layer=collect_probs_per_layer)
                if collect_info and collect_probs_per_layer:
                    logits, _, attn_info_per_layer, logits_per_layer, hidden_states = outputs
                elif collect_info:
                    logits, _, attn_info_per_layer, hidden_states = outputs
                elif collect_probs_per_layer:
                    logits, _, logits_per_layer = outputs
            else:
                logits, _ = self(idx_cond)

            if collect_probs_per_layer:
                logits_per_layer_generated.extend(logits_per_layer)

            logits = logits[:, -1, :] / temperature  # Shape: (1, vocab_size)

            if collect_info and collect_probs_per_layer:
                # Collect next token probabilities per layer
                next_token_probs_per_layer = []
                for layer_logits in logits_per_layer:
                    layer_logits = layer_logits[:, -1, :]  # Shape: (1, vocab_size)
                    layer_probs = F.softmax(layer_logits, dim=-1)

                    # Extract top 10 next token probabilities
                    top_probs, top_indices = torch.topk(layer_probs, k=10, dim=-1)  # Shape: (1, k)
                    next_token_probs_layer = []
                    for idx_token, prob in zip(top_indices[0], top_probs[0]):
                        token_id = int(idx_token.item())
                        probability = float(prob.item())
                        decoded_token = decode([token_id]) if decode else None
                        next_token_probs_layer.append({
                            'token_id': token_id,
                            'decoded_token': decoded_token,
                            'probability': probability
                        })
                    next_token_probs_per_layer.append(next_token_probs_layer)

            # Apply temperature and top_k to final logits
            if top_k is not None:
                current_top_k = min(top_k, logits.size(-1))
                v, _ = torch.topk(logits, k=current_top_k)
                logits[logits < v[:, [-1]]] = -float('inf')

            probs = F.softmax(logits, dim=-1)  # Shape: (1, vocab_size)

            # Extract top 10 next token probabilities before sampling
            top_probs, top_indices = torch.topk(probs, k=10, dim=-1)  # Shape: (1, k)
            next_token_probs = []
            for idx_token, prob in zip(top_indices[0], top_probs[0]):
                token_id = int(idx_token.item())
                probability = float(prob.item())
                decoded_token = decode([token_id]) if decode else None
                next_token_probs.append({
                    'token_id': token_id,
                    'decoded_token': decoded_token,
                    'probability': probability
                })

            # Sample the next token
            idx_next = torch.multinomial(probs, num_samples=1)  # Shape: (1, 1)
            idx = torch.cat((idx, idx_next), dim=1)  # Append to sequence

            if collect_info:
                token_id = idx[:, -2].item()
                # -2 because we are updating information for the token before the last generated one
                decoded_token = decode([token_id]) if decode else None
                token_info = {
                    'token_id': token_id,
                    'decoded_token': decoded_token,
                    'is_initial_context': False,
                    'attn_info_per_layer': [],
                    'next_token_probs': next_token_probs,
                    'most_similar_tokens': [],
                    'next_token_probs_per_layer': next_token_probs_per_layer,  # Add per-layer next token probabilities
                }
                current_token_idx = idx.size(1) - 2  # Index of current token
                token_norm = {
                    'q_norms': [],
                    'k_norms': [],
                    'v_norms': []
                }

                if t > 0:
                    # first one will be duplicate of last initial context token, so only add for t > 0
                    for layer_idx, layer_attn_info in enumerate(attn_info_per_layer):
                        layer_token_info = {}
                        for key in [
                            'q_norms', 'k_norms', 'v_norms', 'embedding_norms',
                            'weighted_v_norms', 'weighted_v_excl_topk_norms',
                            'topk_indices_to', 'topk_values_to', 'topk_indices_from', 'topk_values_from'
                        ]:
                            tensor = layer_attn_info[key]
                            if tensor is not None:
                                layer_token_info[key] = tensor[0, :, -1]  # Shape depends on key
                                # here it's -1 because the attention information is the last one
                            else:
                                layer_token_info[key] = None
                        token_info['attn_info_per_layer'].append(layer_token_info)

                        # Store norms
                        token_norm['q_norms'].append(layer_attn_info['q_norms'][0, :, -1])  # Shape: (nh,)
                        token_norm['k_norms'].append(layer_attn_info['k_norms'][0, :, -1])
                        token_norm['v_norms'].append(layer_attn_info['v_norms'][0, :, -1])

                        # Update attention_scores for tokens attended to by the new token
                        topk_indices = layer_attn_info['topk_indices_to'][0, :, -1]  # Shape: (nh, k)
                        topk_values = layer_attn_info['topk_values_to'][0, :, -1]  # Shape: (nh, k)

                        for head_idx in range(num_heads):
                            indices = topk_indices[head_idx].tolist()
                            values = topk_values[head_idx].tolist()
                            for idx_token, attn_score in zip(indices, values):
                                target_idx = int(idx_token)
                                if target_idx >= idx.size(1) - 1:
                                    continue  # Ignore if index is out of range

                                attention_score = attn_score

                                # Update top tokens attending to the target token
                                top_tokens_dict = dict(
                                    attention_scores[target_idx]['top_tokens_attending_to'][layer_idx][head_idx]
                                )

                                if current_token_idx in top_tokens_dict:
                                    top_tokens_dict[current_token_idx] += attention_score
                                else:
                                    top_tokens_dict[current_token_idx] = attention_score

                                # Keep top K tokens
                                top_k_attn = 5
                                top_k_tokens = heapq.nlargest(top_k_attn, top_tokens_dict.items(), key=lambda x: x[1])
                                attention_scores[target_idx]['top_tokens_attending_to'][layer_idx][
                                    head_idx] = top_k_tokens

                if t > 0:
                    token_norms.append(token_norm)
                    generated_info.append(token_info)

        if collect_info:
            # Include initial context length in the generated_info
            generated_info[0]['initial_context_length'] = initial_context_length
            # Include attention_scores and token_norms in the generated_info
            for idx_info, info in enumerate(generated_info):
                info['attention_scores'] = attention_scores[idx_info]
                info['token_norms'] = token_norms[idx_info]

            # Compute total attention falling on each token per layer and head
            for idx_info, info in enumerate(generated_info):
                total_attention_per_layer_head = []
                for layer_idx in range(num_layers):
                    layer_total_attention = []
                    for head_idx in range(num_heads):
                        total_attention = sum(
                            score for idx_from, score in
                            info['attention_scores']['top_tokens_attending_to'][layer_idx][head_idx]
                        )
                        layer_total_attention.append(total_attention)
                    total_attention_per_layer_head.append(layer_total_attention)
                info['total_attention_per_layer_head'] = total_attention_per_layer_head

            if collect_probs_per_layer:
                return idx, generated_info, logits_per_layer_generated
            else:
                return idx, generated_info
        else:
            if collect_probs_per_layer:
                return idx, logits_per_layer_generated
            else:
                return idx

