import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset.time_series_dataset import TimeSeriesWindows
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


def main() -> None:
    series = lorenz_series()
    dataset = TimeSeriesWindows(series, series, T_in=20, T_out=1)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    d_in = series.shape[1]
    d_model = 64
    d_out = d_in
    core = TimeSeriesHRMCore(d_model, num_heads=4, H_layers=2, L_layers=2)
    model = TimeSeriesHRM(core, d_in=d_in, d_model=d_model, d_out=d_out)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    batch = next(iter(loader))
    loss = ts_train_step(model, batch)
    loss.backward()
    opt.step()
    print(f"Loss after one training step: {loss.item():.2f}")


if __name__ == "__main__":
    main()
