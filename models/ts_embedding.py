import math
from typing import Optional

import torch
import torch.nn as nn


def sinusoidal_pos_encoding(T: int, d: int) -> torch.Tensor:
    """Create standard sinusoidal positional encodings."""
    pe = torch.zeros(T, d)
    position = torch.arange(0, T).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d, 2) * -(math.log(10000.0) / d))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class TimeSeriesEmbedding(nn.Module):
    """Embedding layer for numerical time series.

    Parameters
    ----------
    d_in: int
        Number of input features.
    d_model: int
        Dimension of the model.
    d_time: int, default=32
        Size of optional time features when ``use_sin_time`` is False.
    use_sin_time: bool, default=True
        If True, adds sinusoidal positional encodings; otherwise expects
        ``time_extra`` input that will be linearly projected.
    """

    def __init__(self, d_in: int, d_model: int, d_time: int = 32, use_sin_time: bool = True) -> None:
        super().__init__()
        self.use_sin = use_sin_time
        self.d_model = d_model
        self.time_linear: Optional[nn.Linear] = None
        if not use_sin_time:
            self.time_linear = nn.Linear(d_time, d_model)
            d_in = d_in + d_model
        self.proj = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x: torch.Tensor, time_extra: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [B, T, D_in]
        if self.use_sin:
            B, T, _ = x.shape
            pe = sinusoidal_pos_encoding(T, self.d_model).to(x.device)
            h = self.proj(x)
            return h + pe.unsqueeze(0).expand(B, -1, -1)
        te = self.time_linear(time_extra) if (time_extra is not None and self.time_linear is not None) else 0.0
        x_cat = torch.cat([x, te], dim=-1)
        return self.proj(x_cat)


__all__ = ["TimeSeriesEmbedding", "sinusoidal_pos_encoding"]
