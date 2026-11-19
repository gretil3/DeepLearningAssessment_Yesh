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

# --- Sparse autoencoder (784 -> 256 sparse code h -> 784) ---
SPARSE_HIDDEN_DIM: int = 256
SPARSE_LR: float = 1e-3
SPARSE_EPOCHS: int = 20
# Picked from a short validation-set sweep over {0, 1e-5, 1e-4, 1e-3}: 1e-4 gives clear
# sparsity (~56 active units) without collapse; 1e-3 zeroes almost every unit.
SPARSE_LAMBDA: float = 1e-4  # weight of the L1 penalty on the code h

# --- Variational autoencoder (784 -> 256 -> (mu, logvar) of 64 -> 256 -> 784) ---
VAE_HIDDEN_DIM: int = 256
VAE_LATENT_DIM: int = 64
VAE_LR: float = 1e-3
VAE_EPOCHS: int = 20
VAE_KL_WEIGHT: float = 1.0  # 1.0 is the standard ELBO (no beta-VAE re-weighting)

# --- Vanilla GAN (MLP generator and discriminator, z ~ N(0, I) of size 100) ---
GAN_LATENT_DIM: int = 100
GAN_HIDDEN_DIMS: tuple[int, int, int] = (256, 512, 1024)  # generator widths; the discriminator mirrors them
GAN_DROPOUT: float = 0.3  # discriminator only; keeps it from memorising the real images
GAN_LEAKY_SLOPE: float = 0.2
GAN_LR: float = 2e-4
GAN_BETA1: float = 0.5  # the usual low beta1 for adversarial training
GAN_EPOCHS: int = 50

# --- DCGAN (convolutional generator and discriminator, z ~ N(0, I) of size 100) ---
DCGAN_LATENT_DIM: int = 100
DCGAN_BASE_CHANNELS: int = 64  # generator: 2x -> 1x -> 1 channel; discriminator: 1x -> 2x -> 1 logit
DCGAN_LEAKY_SLOPE: float = 0.2
DCGAN_INIT_STD: float = 0.02  # N(0, 0.02) weight initialisation from the DCGAN guidelines
DCGAN_LR: float = 2e-4  # same optimizer settings as the Vanilla GAN, so only the architecture differs
DCGAN_BETA1: float = 0.5
DCGAN_EPOCHS: int = 50

# --- DDPM (small UNet predicting the noise epsilon, T = 1000) ---
DDPM_TIMESTEPS: int = 1000
DDPM_BETA_START: float = 1e-4  # linear noise schedule of Ho et al. (2020)
DDPM_BETA_END: float = 0.02
DDPM_BASE_CHANNELS: int = 32  # UNet widths: 32 at 28x28, then 64 at 14x14 and 7x7
DDPM_TIME_EMBED_DIM: int = 128
DDPM_GROUPS: int = 8  # GroupNorm groups; must divide every channel count
DDPM_LR: float = 2e-4
DDPM_GRAD_CLIP: float = 1.0  # max gradient norm, as in the DDPM reference implementation
DDPM_EPOCHS: int = 15  # sized for a CPU run (~7 min/epoch); raise it on a GPU such as Colab
DDPM_EMA_DECAY: float = 0.999  # sampling uses the EMA weights

# --- Latent sampling (Step 5 of the MLP experiment) ---
SAMPLING_GRID_SIDE: int = 4  # each figure panel shows a 4x4 grid = 16 images
SAMPLING_NUM_EVAL: int = 1000  # samples used for the quantitative statistics
