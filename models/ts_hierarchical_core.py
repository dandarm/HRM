from __future__ import annotations
from typing import List

import torch
import torch.nn as nn

from .layers import rms_norm, SwiGLU, Attention, RotaryEmbedding


class HRMBlock(nn.Module):
    """Single transformer block used within the hierarchical core."""

    def __init__(self, d_model: int, num_heads: int, expansion: float = 4.0, rms_norm_eps: float = 1e-5) -> None:
        super().__init__()
        head_dim = d_model // num_heads
        self.self_attn = Attention(
            hidden_size=d_model,
            head_dim=head_dim,
            num_heads=num_heads,
            num_key_value_heads=num_heads,
            causal=False,
        )
        self.mlp = SwiGLU(hidden_size=d_model, expansion=expansion)
        self.norm_eps = rms_norm_eps

    def forward(self, cos_sin, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = rms_norm(
            hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states),
            variance_epsilon=self.norm_eps,
        )
        hidden_states = rms_norm(
            hidden_states + self.mlp(hidden_states),
            variance_epsilon=self.norm_eps,
        )
        return hidden_states


class HRMReasoningModule(nn.Module):
    """Repeatedly apply a sequence of blocks with input injection."""

    def __init__(self, layers: List[HRMBlock]) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs) -> torch.Tensor:
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)
        return hidden_states


class TimeSeriesHRMCore(nn.Module):
    """Hierarchical Reasoning core adapted for continuous time series."""

    def __init__(
        self,
        d_model: int,
        *,
        num_heads: int = 8,
        H_layers: int = 2,
        L_layers: int = 2,
        H_cycles: int = 1,
        L_cycles: int = 4,
        rope_theta: float = 10000.0,
        max_position_embeddings: int = 2048,
    ) -> None:
        super().__init__()
        self.H_cycles = H_cycles
        self.L_cycles = L_cycles
        self.rotary_emb = RotaryEmbedding(
            dim=d_model // num_heads,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
        )
        self.H_level = HRMReasoningModule([HRMBlock(d_model, num_heads) for _ in range(H_layers)])
        self.L_level = HRMReasoningModule([HRMBlock(d_model, num_heads) for _ in range(L_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z_H = torch.zeros_like(x)
        z_L = torch.zeros_like(x)
        # Use only the rotary embeddings required for the current sequence length
        cos_sin = self.rotary_emb(x.size(1))

        for H_step in range(self.H_cycles):
            for L_step in range(self.L_cycles):
                if not (H_step == self.H_cycles - 1 and L_step == self.L_cycles - 1):
                    z_L = self.L_level(z_L, z_H + x, cos_sin=cos_sin)
            if H_step != self.H_cycles - 1:
                z_H = self.H_level(z_H, z_L, cos_sin=cos_sin)

        z_L = self.L_level(z_L, z_H + x, cos_sin=cos_sin)
        z_H = self.H_level(z_H, z_L, cos_sin=cos_sin)
        return z_H


__all__ = ["TimeSeriesHRMCore"]
