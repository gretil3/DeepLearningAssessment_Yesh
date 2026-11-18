"""Train the Sparse Autoencoder with an MSE reconstruction loss plus an L1 sparsity penalty.

Run from the repository root: ``python -m train.train_sparse_ae``.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import configs
import data
import metrics
import utils
from models.sparse_ae import SparseAutoencoder
from train import engine

MODEL_NAME = "sparse_ae"


def sparse_loss(model: SparseAutoencoder, images: torch.Tensor) -> torch.Tensor:
    """Reconstruction loss plus the L1 penalty on the code.

    Args:
        model: Sparse autoencoder.
        images: Image batch of shape (B, 1, 28, 28).

    Returns:
        Scalar loss.
    """
    # L = (1 / (B * D)) * sum_i ||x_i - x_hat_i||^2  +  lambda * (1 / B) * sum_i ||h_i||_1
    # where h = ReLU(W x + b) is the code and ||h||_1 = sum_j |h_j|.
    code = model.encode(images)
    reconstruction = F.mse_loss(model.decode(code), images)
    return reconstruction + configs.SPARSE_LAMBDA * code.abs().sum(dim=1).mean()


@torch.no_grad()
def code_statistics(model: SparseAutoencoder, loader: DataLoader) -> dict[str, float]:
    """Measure how sparse the code is on a dataloader.

    Args:
        model: Trained sparse autoencoder (in eval mode).
        loader: Dataloader to measure on (the test loader).

    Returns:
        Dict with the mean L1 norm of h, the fraction of exactly-zero units, and the
        average number of active (non-zero) units per image.
    """
    total_l1, total_zeros, total_items = 0.0, 0, 0
    for images, _ in loader:
        code = model.encode(images.to(configs.DEVICE))
        total_l1 += code.abs().sum().item()
        total_zeros += (code == 0).sum().item()
        total_items += code.size(0)
    code_dim = configs.SPARSE_HIDDEN_DIM
    return {
        "mean_l1_norm": total_l1 / total_items,
        "zero_fraction": total_zeros / (total_items * code_dim),
        "mean_active_units": code_dim - total_zeros / total_items,
    }


def main() -> None:
    """Train the sparse autoencoder, then report test reconstruction and sparsity."""
    data.set_seed()
    train_loader, val_loader, test_loader = data.get_dataloaders()

    model = SparseAutoencoder().to(configs.DEVICE)
    engine.fit(
        model, MODEL_NAME, train_loader, val_loader,
        sparse_loss, configs.SPARSE_LR, configs.SPARSE_EPOCHS,
    )

    model.eval()
    test_metrics = metrics.reconstruction_metrics(model, test_loader)
    sparsity = code_statistics(model, test_loader)
    utils.save_json(
        {
            "model": MODEL_NAME,
            "code_dim": configs.SPARSE_HIDDEN_DIM,
            "lambda": configs.SPARSE_LAMBDA,
            **test_metrics,
            **sparsity,
        },
        configs.RESULT_DIR / f"{MODEL_NAME}.json",
    )
    print(
        f"test mse {test_metrics['mse']:.5f} | psnr {test_metrics['psnr']:.2f} dB | "
        f"ssim {test_metrics['ssim']:.4f} | zero units {sparsity['zero_fraction']:.1%} | "
        f"active units {sparsity['mean_active_units']:.1f}"
    )


if __name__ == "__main__":
    main()
