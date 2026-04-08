# cuhk-mini-CLIP

A CLIP-style image-text alignment system with a systematic 4-factor ablation study analysing the trade-off between efficiency, accuracy, and robustness.

## Project Overview

This project has **two independent tracks**:

| Track | Dataset | Goal |
|-------|---------|------|
| **A — CUHK Buildings** | 200 self-taken CUHK building photos | Image → building name matching |
| **B — Ablation Study** | Flickr30k (31k images, 127k captions) | Find best trade-off: Efficiency × Accuracy × Robustness |

Track B runs a **4-factor × 3-level full factorial experiment** (up to 81 configurations) against a pre-registered baseline (B1), measuring every run on three dimensions: **efficiency** (params, speed), **accuracy** (R@1/5/10), and **robustness** (perturbation tests). All deltas are computed relative to B1.

| Factor | Options |
|--------|---------|
| Image encoder | SmallCNN · **ResNet-18** (B1) · ResNet-50 |
| Text encoder | TextCNN · **BiGRU** (B1) · *(Transformer planned)* |
| Temperature | fix 0.03 · **fix 0.07** (B1) · learnable 0.07 |
| Embed dim | 64 · **128** (B1) · 256 |

## Track B 
### Image encoders:
#### Design choices for SmallCNN

- **4 blocks (not too shallow, not too deep)**: \
   2–3 blocks are often under-capacity; very deep CNNs lose the “lightweight” advantage.
   4 blocks take a mid point.

- **Conv3×3 + BN + ReLU + Pool**: \
   A stable combination: computationally efficient and fast and lightweight(pool in each layer)

- **MLP projection to shared embedding space**: \
   Keeps the same CLIP-style interface as other encoders: output `(B, D)`.

- **Why this matters**: \
   SmallCNN is not “randomly simple.”
   It is a deliberate design to represent the lightweight side of the trade-off space while still preserving representation power.

#### Design choices for ResNet-18 and ResNet-50

- **Why not use pretrained weights?**: \
  The comparison of all three image encoders (SmallCNN / ResNet-18 / ResNet-50) must be fair — SmallCNN has no pretrained option, so neither should the others.

- **Why keep ResNet-18 and ResNet50 when we already have SmallCNN?**: \
  SmallCNN is the lightweight option. \
  ResNet-18 is the middle option — deep enough to learn richer features through resnet, but small enough to train on ~30K images without too much overfitting.  \
  ResNet-50 is the deepest option to learn richest features, but probable to overfit.

### Text encoders:
#### Design choices for TextCNN

- **Why kernel sizes (2, 3, 4, 5)?**: \
  2-gram catches short phrases ("red car", "young woman", which is quite common). \
  3–5-gram catches longer descriptive phrases common in Flickr30k captions. \
  Using all four at the same time lets the model recognize patterns at multiple scales.

- **Why max-pool over time (not average-pool)?**: \
  Max-pool extracts the strongest signal from each filter regardless of where it appears in the sentence.

- **Why word_dim is set to be 128?**: \
  128 is enough since there are not many words in captions. \
  If word_dim is 256/512, then the benefit would be limited however parameters would be far more.

#### Design choices for BiGRU

- **Why use dropout=0.3 if num_dim>1 ?**: \
  If num_dim=1, a warning would emerge. We use dropout=0.3 if num_dim>1 to avoid overfitting.
  

---

## Project Structure

```
mini-CLIP/
├── configs/
│   └── base.yaml               # Training config
│
├── data/
│   └── flickr30k/
│       ├── images/              # 31,784 images
│       ├── train.jsonl          # 127,134 training pairs
│       ├── val.jsonl            # 15,890 validation pairs
│       ├── test.jsonl           # 15,892 test pairs
│       └── split_manifest.json  # Split statistics & hash
│
├── src/
│   ├── train.py                 # Main training loop (CLIP loss, early stopping)
│   ├── evaluate.py              # Evaluation script (TODO)
│   ├── visualize.py             # Visualization utilities (TODO)
│   │
│   ├── models/
│   │   ├── clip_model.py            # CLIP (dual encoder + loss)
│   │   ├── image_encoder_smallcnn.py # 4-block lightweight CNN
│   │   ├── image_encoder_resnet18.py # ResNet-18
│   │   ├── image_encoder_resnet50.py # ResNet-50
│   │   ├── text_encoder_textcnn.py   # Multi-kernel TextCNN
│   │   └── text_encoder_bigru.py     # 2-layer BiGRU + masked mean pool
│   │
│   ├── datasets/
│   │   ├── custom.py              # Dataset class, VocabTable, collate_fn
│   │   ├── convert_flickr30k.py   # Raw Flickr30k → JSONL converter
│   │   └── build_index.py         # Build search index
│   │
│   └── utils/
│       ├── device.py              # Auto-detect device (XPU / CUDA / CPU)
│       └── param_stats.py         # Model parameter statistics
│
├── tests/
│   ├── test_models.py             # Encoder smoke tests (13 tests)
│   └── test_device.py             # Device detection tests (4 tests)
│
├── scripts/
│   └── clean_and_split.py         # Train/val/test split script
│
├── checkpoints/                   # Saved model weights (gitignored)
├── pyproject.toml                 # Project config and dependencies (uv)
├── uv.lock                        # Locked dependency versions
└── .gitignore
```

---

## Setup

### Prerequisites

- Python >= 3.10
- [uv](https://docs.astral.sh/uv/) package manager

### Install & Run

```bash
uv sync                        # one command: creates .venv + installs everything
uv run python -m src.train     # train
uv run pytest                  # test (17 tests)
```

### Switch PyTorch Platform

Default is **CUDA 12.4** (Kaggle / NVIDIA). Edit `pyproject.toml` to switch:

```toml
[[tool.uv.index]]
name = "pytorch"
url = "https://download.pytorch.org/whl/cu124"     # CUDA (default)
# url = "https://pytorch-extension.intel.com/release-whl/stable/xpu/us/"  # Intel XPU
# url = "https://download.pytorch.org/whl/cpu"      # CPU only
```

Then `uv sync` again.

---

## Config

All experiment settings in `configs/base.yaml`:

```yaml
model:
  image_encoder: resnet18  # smallcnn | resnet18 | resnet50
  text_encoder: bigru  # textcnn | bigru
  embed_dim: 128  # 64 | 128 | 256

train:
  temperature: 0.07
  temperature_mode: fixed  # fixed | learn
  batch_size: 32
  epochs: 20
  patience: 5  # early stopping: stop if val_loss doesn't improve for N=patience epochs
```

