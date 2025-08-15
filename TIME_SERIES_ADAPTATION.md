# Time Series Extension for HRM

This repository has been extended to handle time series forecasting tasks. The main additions introduce a dataset helper, a flexible input embedding, a regression head, and an adapter that connects these components with the core Hierarchical Reasoning Model (HRM).

## `dataset/time_series_dataset.py`
* **Purpose:** Generate sliding windows of normalized input and target sequences.
* **Key Features:**
  - Computes per-feature mean and standard deviation, applying z-score normalization.
  - Supports optional calendar-based extra features that are stacked alongside the sequence.
  - Returns PyTorch tensors with keys `x_in`, `y_out`, and optional `time_extra`.

## `models/ts_embedding.py`
* **Purpose:** Map numerical sequences into a latent space.
* **Key Features:**
  - Offers sinusoidal positional encoding or linear projection of user-provided temporal features.
  - Uses LayerNorm and a two-layer feed-forward block to project inputs to `d_model` dimensions.

## `models/ts_head.py`
* **Purpose:** Produce predictions and compute regression losses.
* **Key Features:**
  - Can output direct values or Gaussian parameters (`mu`, `log_sigma`).
  - Provides Mean Squared Error, Huber loss, or Gaussian negative log-likelihood.

## `models/ts_hrm_adapter.py`
* **Purpose:** Bridge the time series modules with a pre-existing HRM core.
* **Key Features:**
  - Embeds inputs, forwards them through the HRM, and applies the regression head.
  - Includes a helper `ts_train_step` that performs a forward pass and returns the appropriate loss.

These components together allow the HRM architecture to model temporal dynamics, capturing both short-term variations and long-range trends within sequential data.
