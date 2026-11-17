"""Central configuration: every hyperparameter and path lives here."""

from pathlib import Path

import torch

# --- Reproducibility ---
SEED: int = 42

# --- Paths (anchored to this file so scripts work from any cwd, locally or on Colab) ---
ROOT_DIR: Path = Path(__file__).resolve().parent
DATA_DIR: Path = ROOT_DIR / "data"
OUTPUT_DIR: Path = ROOT_DIR / "outputs"
CHECKPOINT_DIR: Path = OUTPUT_DIR / "checkpoints"
LOG_DIR: Path = OUTPUT_DIR / "logs"
FIGURE_DIR: Path = OUTPUT_DIR / "figures"
RESULT_DIR: Path = OUTPUT_DIR / "results"

# --- Device ---
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Data ---
BATCH_SIZE: int = 128
# Train/val/test fractions applied to the merged 70k Fashion-MNIST samples.
SPLIT_RATIOS: tuple[float, float, float] = (0.8, 0.1, 0.1)
# 0 avoids multiprocessing issues on Windows and in notebooks; data is small enough.
NUM_WORKERS: int = 0
