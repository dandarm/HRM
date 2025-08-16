# tests/test_time_series_windows.py
#import pytest
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]  # .../HRM
sys.path.insert(0, str(ROOT))

from dataset import TimeSeriesWindows, HAS_NUMPY

#@pytest.mark.parametrize("T_in,T_out", [(16, 8), (32, 16)])
def test_time_series_windows_shapes(toy_ts_batch, T_in, T_out):
    X, Y = toy_ts_batch["X"], toy_ts_batch["Y"]

    # Se HAS_NUMPY è True, ci aspettiamo che l'implementazione NumPy sia stata selezionata
    # automaticamente dal router in dataset/__init__.py
    ds = TimeSeriesWindows(X, Y, T_in=T_in, T_out=T_out, stride=1)

    assert len(ds) == (len(X) - T_in - T_out + 1) // 1 + int((len(X) - T_in - T_out + 1) % 1 == 0)
    sample = ds[0]
    assert "x_in" in sample and "y_out" in sample
    assert sample["x_in"].shape[-2:] == (T_in, X.shape[-1])
    assert sample["y_out"].shape[-2:] == (T_out, Y.shape[-1])

    # Nessun errore/avviso del tipo “NumPy non installabile…” deve emergere qui
    # perché il router ha già scelto la classe giusta.
