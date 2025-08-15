from typing import Optional, Dict

import torch
import torch.nn as nn

from .ts_embedding import TimeSeriesEmbedding
from .ts_head import TSRegressionHead, regression_loss


class TimeSeriesHRM(nn.Module):
    """Wrapper that adapts an HRM core to time series forecasting."""

    def __init__(self, hrm_core: nn.Module, d_in: int, d_model: int, d_out: int, gaussian: bool = False) -> None:
        super().__init__()
        self.hrm = hrm_core
        self.embed = TimeSeriesEmbedding(d_in=d_in, d_model=d_model)
        self.head = TSRegressionHead(d_model=d_model, d_out=d_out, gaussian=gaussian)

    def forward(
        self,
        x_in: torch.Tensor,
        *,
        time_extra: Optional[torch.Tensor] = None,
        T_out: int = 1,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass through embedding, HRM core and head.

        Parameters
        ----------
        x_in: torch.Tensor
            Input sequence ``[B, T_in, D_in]``.
        time_extra: Optional[torch.Tensor]
            Additional time encodings ``[B, T_in + T_out, d_time]`` when
            the embedding is configured with ``use_sin_time=False``.
        T_out: int
            Number of forecast steps to return.
        """

        h0 = self.embed(x_in, time_extra=time_extra)
        h_out = self.hrm(h0)
        y_pred, log_sigma = self.head(h_out[:, -T_out:, :])
        return y_pred, log_sigma


def ts_train_step(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    *,
    gaussian: bool = False,
    huber: bool = False,
) -> torch.Tensor:
    x = batch["x_in"]
    y = batch["y_out"]
    time_extra = batch.get("time_extra")
    y_hat, log_sigma = model(x, time_extra=time_extra, T_out=y.shape[1])
    loss = regression_loss(y_hat, y, gaussian=gaussian, log_sigma=log_sigma, huber=huber)
    return loss


__all__ = ["TimeSeriesHRM", "ts_train_step"]
