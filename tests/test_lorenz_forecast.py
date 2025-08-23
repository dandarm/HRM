from pathlib import Path
import numpy as np
import torch
device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch import nn
from torch.utils.data import DataLoader
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - plotting is optional in tests
    plt = None

import sys
ROOT = Path(__file__).resolve().parents[1]  # .../HRM
sys.path.insert(0, str(ROOT))

from dataset import TimeSeriesWindows
from models.ts_hierarchical_core import TimeSeriesHRMCore
from models.ts_hrm_adapter import TimeSeriesHRM, ts_train_step


def lorenz_series(T: int = 1000, dt: float = 0.01,
                  sigma: float = 10.0, rho: float = 28.0,
                  beta: float = 8/3) -> np.ndarray:
    """Generate a Lorenz attractor time series of length ``T``.

    Parameters
    ----------
    T: int
        Number of timesteps to simulate.
    dt: float
        Integration step size.
    sigma, rho, beta: float
        Standard Lorenz system parameters.
    """
    xyz = np.zeros((T, 3), dtype=np.float32)
    xyz[0] = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    for t in range(1, T):
        x, y, z = xyz[t - 1]
        x_dot = sigma * (y - x)
        y_dot = x * (rho - z) - y
        z_dot = x * y - beta * z
        xyz[t] = xyz[t - 1] + dt * np.array([x_dot, y_dot, z_dot], dtype=np.float32)
    return xyz

# Non la usiamo più
class GRUCore(nn.Module):
    """Minimal recurrent core to plug into :class:`TimeSeriesHRM`."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.rnn = nn.GRU(d_model, d_model, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return out

      
def main() -> None:
    series = lorenz_series()
    dataset = TimeSeriesWindows(series, series, T_in=20, T_out=1)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    d_in = series.shape[1]
    d_model = 64
    d_out = d_in

    core = TimeSeriesHRMCore(d_model, num_heads=4, H_layers=2, L_layers=2)
    model = TimeSeriesHRM(core, d_in=d_in, d_model=d_model, d_out=d_out).to(device)

    # model = TimeSeriesHRM(GRUCore(d_model), d_in=d_in, d_model=d_model, d_out=d_out)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 5
    for epoch in range(num_epochs):
        for batch in loader:
            for k, v in list(batch.items()):
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device, non_blocking=True)
            opt.zero_grad()
            loss = ts_train_step(model, batch)
            loss.backward()
            opt.step()
        print(f"Epoch {epoch+1}/{num_epochs}, loss: {loss.item():.2f}")

    # Inference
    mu = torch.tensor(dataset.stats["mu"], dtype=torch.float32, device=device)
    sd = torch.tensor(dataset.stats["sd"], dtype=torch.float32, device=device)
    x_win = ((torch.tensor(series[: dataset.T_in], dtype=torch.float32, device=device) - mu) / sd).unsqueeze(0)
    preds = []
    with torch.no_grad():
        for _ in range(dataset.T_in, len(series)):
            y_hat, _ = model(x_win)
            y_pred = y_hat[:, -1, :]
            preds.append(y_pred.squeeze(0).cpu().numpy())
            y_norm = (y_pred - mu) / sd
            x_win = torch.cat([x_win[:, 1:, :], y_norm.unsqueeze(1)], dim=1)
    preds = np.stack(preds)

    # Plot ground truth and forecast for first dimension
    if plt is not None:
        plt.figure()
        plt.plot(series[:, 0], label="Ground Truth")
        plt.plot(range(dataset.T_in, len(series)), preds[:, 0], label="Forecast")
        plt.xlabel("Time")
        plt.ylabel("x")
        plt.legend()
        plt.tight_layout()
        plt.savefig("lorenz_forecast.png")
        print("Saved plot to lorenz_forecast.png")
    else:  # pragma: no cover - only executed when matplotlib missing
        print("Matplotlib not installed, skipping plot.")


if __name__ == "__main__":
    main()
