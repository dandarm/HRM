import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Callable, Dict, Any


class TimeSeriesWindows(Dataset):
    """Dataset that slices continuous time series into input/output windows.

    Parameters
    ----------
    X: np.ndarray
        Array of input features with shape ``[N, D_in]`` ordered by time.
    Y: np.ndarray
        Array of targets with shape ``[N, D_out]``.
    T_in: int
        Number of timesteps to include in the input window.
    T_out: int
        Number of timesteps to predict.
    stride: int, default=1
        Step between consecutive windows.
    stats: Optional[Dict[str, np.ndarray]], default=None
        Pre-computed normalization statistics.  If ``None`` they will be
        computed from ``X`` as ``mu`` and ``sd``.
    calendar_feats: Optional[Callable[[int], Any]], default=None
        Function returning additional temporal features for a timestep index.
    """

    def __init__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        T_in: int,
        T_out: int,
        stride: int = 1,
        stats: Optional[Dict[str, np.ndarray]] = None,
        calendar_feats: Optional[Callable[[int], Any]] = None,
    ) -> None:
        super().__init__()
        self.X = X
        self.Y = Y
        self.T_in = T_in
        self.T_out = T_out
        self.stride = stride
        self.calendar = calendar_feats

        if stats is None:
            mu = X.mean(0, keepdims=True)
            sd = X.std(0, keepdims=True) + 1e-6
            stats = {"mu": mu, "sd": sd}
        self.stats = stats

        self.idxs = [t for t in range(T_in, len(X) - T_out + 1, stride)]

    def __len__(self) -> int:
        return len(self.idxs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        t = self.idxs[idx]
        x_win = (self.X[t - self.T_in : t] - self.stats["mu"]) / self.stats["sd"]
        y_win = self.Y[t : t + self.T_out]

        time_extra = None
        if self.calendar is not None:
            feats = [self.calendar(k) for k in range(t - self.T_in, t + self.T_out)]
            time_extra = np.stack(feats, 0)

        item = {
            "x_in": torch.tensor(x_win, dtype=torch.float32),
            "y_out": torch.tensor(y_win, dtype=torch.float32),
        }
        if time_extra is not None:
            item["time_extra"] = torch.tensor(time_extra, dtype=torch.float32)
        return item


__all__ = ["TimeSeriesWindows"]
