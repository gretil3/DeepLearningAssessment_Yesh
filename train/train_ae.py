"""Train the Vanilla Autoencoder with an MSE reconstruction loss.

Run from the repository root: ``python -m train.train_ae``.
"""

import torch
import torch.nn.functional as F
from torch import nn

import configs
import data
import metrics
import utils
from models.vanilla_ae import VanillaAutoencoder
from train import engine

MODEL_NAME = "vanilla_ae"


def reconstruction_loss(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Mean-squared-error reconstruction loss of a batch.

    Args:
        model: Autoencoder-like model mapping images to reconstructions.
        images: Image batch of shape (B, 1, 28, 28).

    Returns:
        Scalar loss.
    """
    # MSE: L = (1 / (B * D)) * sum_i ||x_i - x_hat_i||^2, with x_hat = Decoder(Encoder(x))
    return F.mse_loss(model(images), images)


def main() -> None:
    """Train the autoencoder, then report test reconstruction metrics."""
    data.set_seed()
    train_loader, val_loader, test_loader = data.get_dataloaders()

    model = VanillaAutoencoder().to(configs.DEVICE)
    engine.fit(
        model, MODEL_NAME, train_loader, val_loader,
        reconstruction_loss, configs.AE_LR, configs.AE_EPOCHS,
    )

    model.eval()
    test_metrics = metrics.reconstruction_metrics(model, test_loader)
    utils.save_json(
        {"model": MODEL_NAME, "latent_dim": configs.AE_LATENT_DIM, **test_metrics},
        configs.RESULT_DIR / f"{MODEL_NAME}.json",
    )
    print(
        f"test mse {test_metrics['mse']:.5f} | psnr {test_metrics['psnr']:.2f} dB | "
        f"ssim {test_metrics['ssim']:.4f}"
    )


if __name__ == "__main__":
    main()
