
# HRM for Time Series

**Goals: bring HRM’s latent hierarchical reasoning into the time‑series domain**

**Repository changes: added a time‑series adapter (`models/ts_hrm_adapter.py: TimeSeriesHRM`), input embedding (`models/ts_embedding.py: TimeSeriesEmbedding`), regression head (`models/ts_head.py: TSRegressionHead`), sliding‑window datasets with a transparent router (`dataset/__init__.py: TimeSeriesWindows`, backed by `time_series_dataset_np.py` / `time_series_dataset_torch.py`), and dynamic attention backend selection with logging (`models/layers.py: set_use_flash`, logger `hrm.attn`).**

* **Data pipeline:** windowing **T\_in → T\_out** with normalization and optional calendar features via **`dataset.TimeSeriesWindows`** (NumPy or PyTorch backend selected by `dataset/__init__.py`). Outputs are PyTorch tensors ready for training.
* **Temporal embedding:** project features to `d_model` with time encodings (sinusoidal and/or calendar) using **`models.ts_embedding.TimeSeriesEmbedding`**.
* **Integration with the HRM core:** the L↔H core remains unchanged and is orchestrated by **`models.ts_hrm_adapter.TimeSeriesHRM`**; the output is produced by **`models.ts_head.TSRegressionHead`** in either one‑shot multi‑horizon or autoregressive mode.
* **Rotary Position Embeddings (RoPE):** index `cos/sin` at actual positions (or `position_ids`) and reshape for correct broadcasting in **`models.layers.apply_rotary_pos_emb`**; this generalizes to windows shorter than the maximum length.



# Hierarchical Reasoning Model

![](./assets/hrm.png)

Reasoning, the process of devising and executing complex goal-oriented action sequences, remains a critical challenge in AI.
Current large language models (LLMs) primarily employ Chain-of-Thought (CoT) techniques, which suffer from brittle task decomposition, extensive data requirements, and high latency. Inspired by the hierarchical and multi-timescale processing in the human brain, we propose the Hierarchical Reasoning Model (HRM), a novel recurrent architecture that attains significant computational depth while maintaining both training stability and efficiency.
HRM executes sequential reasoning tasks in a single forward pass without explicit supervision of the intermediate process, through two interdependent recurrent modules: a high-level module responsible for slow, abstract planning, and a low-level module handling rapid, detailed computations. With only 27 million parameters, HRM achieves exceptional performance on complex reasoning tasks using only 1000 training samples. The model operates without pre-training or CoT data, yet achieves nearly perfect performance on challenging tasks including complex Sudoku puzzles and optimal path finding in large mazes.
Furthermore, HRM outperforms much larger models with significantly longer context windows on the Abstraction and Reasoning Corpus (ARC), a key benchmark for measuring artificial general intelligence capabilities.
These results underscore HRM’s potential as a transformative advancement toward universal computation and general-purpose reasoning systems.

## Quick Start Guide 🚀

### Prerequisites ⚙️

Ensure PyTorch and CUDA are installed. The repo needs CUDA extensions to be built. If not present, run the following commands:

```bash
# Install CUDA 12.6
CUDA_URL=https://developer.download.nvidia.com/compute/cuda/12.6.3/local_installers/cuda_12.6.3_560.35.05_linux.run

wget -q --show-progress --progress=bar:force:noscroll -O cuda_installer.run $CUDA_URL
sudo sh cuda_installer.run --silent --toolkit --override

export CUDA_HOME=/usr/local/cuda-12.6

# Install PyTorch with CUDA 12.6
PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu126

pip3 install torch torchvision torchaudio --index-url $PYTORCH_INDEX_URL

# Additional packages for building extensions
pip3 install packaging ninja wheel setuptools setuptools-scm
```

Then install FlashAttention. For Hopper GPUs, install FlashAttention 3

```bash
git clone git@github.com:Dao-AILab/flash-attention.git
cd flash-attention/hopper
python setup.py install
```

For Ampere or earlier GPUs, install FlashAttention 2

```bash
pip3 install flash-attn
```

## Install Python Dependencies 🐍

```bash
pip install -r requirements.txt
```


## Citation 📜

```bibtex
@misc{wang2025hierarchicalreasoningmodel,
      title={Hierarchical Reasoning Model}, 
      author={Guan Wang and Jin Li and Yuhao Sun and Xing Chen and Changling Liu and Yue Wu and Meng Lu and Sen Song and Yasin Abbasi Yadkori},
      year={2025},
      eprint={2506.21734},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2506.21734}, 
}
```