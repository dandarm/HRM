import os

from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F

try:
    from flash_attn.flash_attn_interface import flash_attn_func  # type: ignore[import]
    _HAS_FLASH = True
except ImportError:
    # Fallback to FlashAttention 2
    #from flash_attn import flash_attn_func  # type: ignore[import]
    # in case of 2080Ti - older than Ampere GPUs
    flash_attn_func = None
    _HAS_FLASH = False

def _sm_ge(device, major, minor=0):
    if not torch.cuda.is_available():
        return False
    m, n = torch.cuda.get_device_capability(device)
    return (m > major) or (m == major and n >= minor)
_USE_FLASH_ENV = os.environ.get("USE_FLASH", "1") == "1"

from models.common import trunc_normal_init_


CosSin = Tuple[torch.Tensor, torch.Tensor]


def _find_multiple(a, b):
    return (-(a // -b)) * b


# def rotate_half(x: torch.Tensor):
#     """Rotates half the hidden dims of the input."""
#     x1 = x[..., : x.shape[-1] // 2]
#     x2 = x[..., x.shape[-1] // 2 :]
#     return torch.cat((-x2, x1), dim=-1)
def rotate_half(x):
    # x: [..., Dh] con Dh pari
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).reshape(*x.shape)


# def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
#     # q, k: [bs, seq_len, num_heads, head_dim]
#     # cos, sin: [seq_len, head_dim]
#     orig_dtype = q.dtype
#     q = q.to(cos.dtype)
#     k = k.to(cos.dtype)

#     q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
#     k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))

#     return q_embed.to(orig_dtype), k_embed.to(orig_dtype)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    """
    q, k: [B, S, H, Dh]  (assunto coerente con il tuo stack)
    cos, sin: [Lmax, Dh]  (cache precomputed)
    position_ids: [S] o [B,S] opzionale; se None -> arange(0, S)

    Restituisce: q_embed, k_embed con stessa shape di q/k.
    """
    B, S, H, Dh = q.shape
    device = q.device
    dtype = q.dtype

    # costruisci gli indici di posizione per questa finestra
    if position_ids is None:
        pos = torch.arange(S, device=device)  # [S]
    else:
        # consenti sia [S] che [B,S]; in caso [B,S] prendi la prima riga (omogeneo per batch)
        pos = position_ids
        if pos.dim() == 2:
            pos = pos[0]
        assert pos.numel() == S, f"position_ids size {pos.numel()} != seq_len {S}"

    # slice dei tensori RoPE *sulle posizioni effettive*
    # cos_sin: [S, Dh]
    cos_s = cos.index_select(0, pos).to(device=device, dtype=dtype)
    sin_s = sin.index_select(0, pos).to(device=device, dtype=dtype)

    # reshape per broadcast con q/k [B, S, H, Dh]
    # -> [1, S, 1, Dh]
    cos_s = cos_s.unsqueeze(0).unsqueeze(2)
    sin_s = sin_s.unsqueeze(0).unsqueeze(2)

    # applica RoPE
    q_embed = (q * cos_s) + (rotate_half(q) * sin_s)
    k_embed = (k * cos_s) + (rotate_half(k) * sin_s)
    return q_embed, k_embed


class CastedLinear(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool):
        super().__init__()
        # Truncated LeCun normal init
        self.weight = nn.Parameter(
            trunc_normal_init_(torch.empty((out_features, in_features)), std=1.0 / (in_features ** 0.5))
        )
        self.bias = None
        if bias:
            # Zero init bias
            self.bias = nn.Parameter(torch.zeros((out_features, )))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)


class CastedEmbedding(nn.Module):
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 init_std: float,
                 cast_to: torch.dtype):
        super().__init__()
        self.cast_to = cast_to

        # Truncated LeCun normal init
        self.embedding_weight = nn.Parameter(
            trunc_normal_init_(torch.empty((num_embeddings, embedding_dim)), std=init_std)
        )
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.embedding(input, self.embedding_weight.to(self.cast_to))


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings, base, device=None):
        super().__init__()

        # RoPE
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)

        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = nn.Buffer(emb.cos(), persistent=False)
        self.sin_cached = nn.Buffer(emb.sin(), persistent=False)

    def forward(self):
        return self.cos_cached, self.sin_cached


class Attention(nn.Module):
    def __init__(self, hidden_size, head_dim, num_heads, num_key_value_heads, causal=False):
        super().__init__()

        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.output_size = head_dim * num_heads
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.causal = causal

        self.qkv_proj = CastedLinear(self.hidden_size, (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim, bias=False)
        self.o_proj = CastedLinear(self.output_size, self.hidden_size, bias=False)

    # def use_flash():
    #     return (
    #         _USE_FLASH_ENV
    #         and _HAS_FLASH
    #         #and query.is_cuda and key.is_cuda and value.is_cuda
    #         and _sm_ge(query.device, 8, 0)                    # Ampere+
    #         and query.dtype in (torch.float16, torch.bfloat16) # tipico per flash-attn
    #         )

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        # hidden_states: [bs, seq_len, num_heads, head_dim]
        qkv = self.qkv_proj(hidden_states)

        # Split head
        qkv = qkv.view(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        query = qkv[:, :, :self.num_heads]
        key = qkv[:, :, self.num_heads: self.num_heads + self.num_key_value_heads]
        value = qkv[:, :, self.num_heads + self.num_key_value_heads:]

        # RoPE
        if cos_sin is not None:
            cos, sin = cos_sin
            offset = 0
            position_ids = torch.arange(offset, offset + seq_len, device=hidden_states.device)
            query, key = apply_rotary_pos_emb(query, key, cos, sin, position_ids)

        # flash attn
        #attn_output = flash_attn_func(q=query, k=key, v=value, causal=self.causal)

        # Fallback: SDPA di PyTorch
        # SDPA usa [B, H, S, Dh] => permutiamo e poi torniamo indietro
        q = query.transpose(1, 2)   # [B,H,S,Dh]
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)
        attn_output = attn.transpose(1, 2)  # [B,S,H,Dh]

        if isinstance(attn_output, tuple):  # fa2 and fa3 compatibility
            attn_output = attn_output[0]

        # attn_output: [batch_size, num_heads, seq_len, head_dim]
        attn_output = attn_output.view(batch_size, seq_len, self.output_size)  # type: ignore
        return self.o_proj(attn_output)


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, expansion: float):
        super().__init__()
        inter = _find_multiple(round(expansion * hidden_size * 2 / 3), 256)

        self.gate_up_proj = CastedLinear(hidden_size, inter * 2, bias=False)
        self.down_proj    = CastedLinear(inter, hidden_size, bias=False)

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


def rms_norm(hidden_states: torch.Tensor, variance_epsilon: float) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)

    variance = hidden_states.square().mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
    return hidden_states.to(input_dtype)
