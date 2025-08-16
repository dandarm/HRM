# tests/test_time_series_windows.py
"""Unit tests for the :class:`TimeSeriesWindows` dataset.

These tests provide a minimal synthetic time-series batch to ensure that the
dataset slices windows with the expected shapes.  The original test template in
this repository lacked fixtures and parametrization; this version restores the
intended behaviour.
"""

from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]  # .../HRM
sys.path.insert(0, str(ROOT))

from dataset import TimeSeriesWindows, HAS_NUMPY


@pytest.fixture
def toy_ts_batch():
    """Return a simple synthetic time-series batch for testing."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 3)).astype(np.float32)
    Y = rng.standard_normal((100, 3)).astype(np.float32)
    return {"X": X, "Y": Y}


@pytest.mark.parametrize("T_in,T_out", [(16, 8), (32, 16)])
def test_time_series_windows_shapes(toy_ts_batch, T_in, T_out):
    X, Y = toy_ts_batch["X"], toy_ts_batch["Y"]

    # If NumPy is available, the router in dataset/__init__.py should select the
    # NumPy implementation automatically.
    ds = TimeSeriesWindows(X, Y, T_in=T_in, T_out=T_out, stride=1)

    assert len(ds) == (len(X) - T_in - T_out + 1)
    sample = ds[0]
    assert "x_in" in sample and "y_out" in sample
    assert sample["x_in"].shape[-2:] == (T_in, X.shape[-1])
    assert sample["y_out"].shape[-2:] == (T_out, Y.shape[-1])

    # No warnings about missing NumPy should surface here because the router has
    # already selected the correct backend.

