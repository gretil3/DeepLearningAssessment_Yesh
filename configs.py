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
IMAGE_SIZE: int = 28
INPUT_DIM: int = IMAGE_SIZE * IMAGE_SIZE  # flattened Fashion-MNIST image
NUM_CLASSES: int = 10
CLASS_NAMES: tuple[str, ...] = (
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
)

# --- Figures ---
FIGURE_DPI: int = 300  # the report requires high-resolution figures

# --- MLP classifier (784 -> 256 -> 64 -> 10) ---
MLP_HIDDEN_DIMS: tuple[int, int] = (256, 64)
MLP_LR: float = 1e-3
MLP_EPOCHS: int = 20
MLP_TARGET_ACC: float = 0.88  # minimum test accuracy required by the assignment

# --- Vanilla autoencoder (784 -> 256 -> 64 -> 256 -> 784) ---
# The frozen-latent decoder experiment reuses these values so that only the encoder differs.
AE_HIDDEN_DIM: int = 256
AE_LATENT_DIM: int = 64
AE_LR: float = 1e-3
AE_EPOCHS: int = 20
