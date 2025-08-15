from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TSRegressionHead(nn.Module):
    """Simple regression head for time series forecasting."""

    def __init__(self, d_model: int, d_out: int, gaussian: bool = False) -> None:
        super().__init__()
        self.gaussian = gaussian
        out_dim = d_out if not gaussian else 2 * d_out
        self.proj = nn.Linear(d_model, out_dim)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        out = self.proj(h)
        if not self.gaussian:
            return out, None
        d = out.shape[-1] // 2
        mu = out[..., :d]
        log_sigma = out[..., d:].clamp(-5, 5)
        return mu, log_sigma


def gaussian_nll(mu: torch.Tensor, log_sigma: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    var = torch.exp(2 * log_sigma)
    return 0.5 * torch.log(2 * torch.pi * var) + (y - mu) ** 2 / (2 * var)


def regression_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    gaussian: bool = False,
    log_sigma: Optional[torch.Tensor] = None,
    huber: bool = False,
    delta: float = 1.0,
) -> torch.Tensor:
    if gaussian:
        assert log_sigma is not None
        return gaussian_nll(pred, log_sigma, target).mean()
    if huber:
        return F.huber_loss(pred, target, delta=delta)
    return F.mse_loss(pred, target)


__all__ = ["TSRegressionHead", "regression_loss"]
