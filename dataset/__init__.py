# dataset/__init__.py
import importlib
import importlib.util

HAS_NUMPY = importlib.util.find_spec("numpy") is not None

_np_impls = [
    ".time_series_dataset_np",
    ".time_series_dataset_numpy",
    ".time_series_dataset",  # se la versione "originale" NumPy è in questo file
]
_torch_impls = [
    ".time_series_dataset_torch",
    ".time_series_dataset_pt",
]

def _try_import(candidates):
    for mod in candidates:
        try:
            return importlib.import_module(mod, package=__name__)
        except Exception:
            continue
    return None

_impl = None
if HAS_NUMPY:
    _impl = _try_import(_np_impls) or _try_import(_torch_impls)
else:
    _impl = _try_import(_torch_impls) or _try_import(_np_impls)

if _impl is None:
    raise ImportError(
        "Impossibile trovare un'implementazione per TimeSeriesWindows "
        "(attesi file: time_series_dataset_np.py / time_series_dataset_torch.py / time_series_dataset.py)."
    )

TimeSeriesWindows = getattr(_impl, "TimeSeriesWindows")

__all__ = ["TimeSeriesWindows", "HAS_NUMPY"]
