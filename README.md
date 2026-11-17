# Comparative Study of Generative Models (Deep Learning D6421, Session 07)

A PyTorch study that compares six generative models on Fashion-MNIST: Vanilla
Autoencoder, Sparse Autoencoder, VAE, Vanilla GAN, DCGAN, and DDPM. An MLP
classifier is also trained to show why discriminative features cannot be used
directly for image generation.

All models share the same data split, batch size, and random seed (42), and every
metric is computed on the held-out test set.

> Work in progress: modules are added one at a time. This README is updated with each one.

## Installation

Requires Python 3.10+.

```bash
python -m venv .venv
.venv\Scripts\activate        # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Fashion-MNIST is downloaded automatically by `torchvision` into `./data` on first use.
The code picks CUDA when available and falls back to CPU, so it runs locally and on Colab.

## Usage

Quick check of the data pipeline (prints the size of each split):

```bash
python -c "import data; tr, va, te = data.get_dataloaders(); print(len(tr.dataset), len(va.dataset), len(te.dataset))"
```

Expected output: `56000 7000 7000`.

Training scripts are run as modules from the repository root. Each script saves its
checkpoint, per-epoch loss, and per-epoch training time.

| Command | What it does | Outputs |
|---|---|---|
| `python -m train.train_mlp` | Trains the 784→256→64→10 MLP classifier (target test accuracy ≥ 88%) | `outputs/checkpoints/mlp_classifier.pt`, `outputs/logs/mlp_classifier.json`, `outputs/results/mlp_classifier.json` |

Experiments are run the same way and read the trained checkpoints, so run the matching
training script first.

| Command | What it does | Outputs |
|---|---|---|
| `python -m experiments.mlp_class_projection` | Step 2: back-projects one-hot class vectors to pixels with transposed MLP weights (no training) and measures weight orthogonality | `outputs/figures/mlp_class_projection.png`, `outputs/results/mlp_class_projection.json` |

Each script is listed here once it is added.

## Project structure

| Path | Description |
|---|---|
| `configs.py` | Single source of seed, paths, device, batch size, and split ratios |
| `data.py` | `set_seed()` and `get_dataloaders()` (merged 70k samples, 80/10/10 split) |
| `utils.py` | `Timer`, JSON helper, `TrainingHistory` (loss and time per epoch), checkpoint save/load, `save_figure` |
| `models/mlp_classifier.py` | Bias-free MLP classifier with an `encode()` method returning the 64-d latent feature |
| `train/train_mlp.py` | Training script for the MLP classifier (keeps the best epoch by validation loss) |
| `experiments/mlp_class_projection.py` | Step 2 of the MLP experiment: class-vector back-projection through W^T and orthogonality check |
| `requirements.txt` | Pinned dependencies |
| `data/` | Downloaded dataset (git-ignored) |
| `outputs/checkpoints/` | Model weights (git-ignored) |
| `outputs/logs/` | Training logs as JSON |
| `outputs/figures/` | Generated figures |
| `outputs/results/` | Metric tables (CSV/JSON) |

`outputs/` subfolders are created on first write.
