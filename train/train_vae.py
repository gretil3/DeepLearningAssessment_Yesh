"""Train the VAE by minimising the negative ELBO.

Run from the repository root: ``python -m train.train_vae``.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import configs
import data
import metrics
import utils
from models.vae import VAE
from train import engine

MODEL_NAME = "vae"


def vae_loss(model: VAE, images: torch.Tensor) -> torch.Tensor:
    """Negative ELBO of a batch, averaged over the images.

    Args:
        model: Variational autoencoder.
        images: Image batch of shape (B, 1, 28, 28) with values in [0, 1].

    Returns:
        Scalar loss.
    """
    # -ELBO_i = BCE_i + w * KL_i, averaged over the batch, with
    #   BCE_i = -sum_d [ x_d log(x_hat_d) + (1 - x_d) log(1 - x_hat_d) ]
    #   KL_i  = KL( N(mu_i, diag(sigma_i^2)) || N(0, I) )
    #         = -1/2 * sum_j (1 + log sigma_ij^2 - mu_ij^2 - sigma_ij^2)
    reconstruction, mu, logvar = model(images)
    batch_size = images.size(0)
    bce = F.binary_cross_entropy(reconstruction, images, reduction="sum") / batch_size
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
    return bce + configs.VAE_KL_WEIGHT * kl


@torch.no_grad()
def posterior_statistics(model: VAE, loader: DataLoader) -> dict[str, float]:
    """Measure the KL term and how many latent dimensions the posterior actually uses.

    A dimension whose mean KL is near zero has collapsed onto the prior and carries no
    information about the image.

    Args:
        model: Trained VAE (in eval mode).
        loader: Dataloader to measure on (the test loader).

    Returns:
        Dict with the mean KL per image (nats) and the number of dimensions with a
        mean KL above 0.01 nats.
    """
    kl_per_dim = torch.zeros(configs.VAE_LATENT_DIM, device=configs.DEVICE)
    total_items = 0
    for images, _ in loader:
        mu, logvar = model.encode(images.to(configs.DEVICE))
        kl_per_dim += (0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar)).sum(dim=0)
        total_items += images.size(0)
    kl_per_dim /= total_items
    return {
        "mean_kl_per_image": kl_per_dim.sum().item(),
        "active_latent_dims": int((kl_per_dim > 0.01).sum().item()),
    }


def main() -> None:
    """Train the VAE, then report test negative ELBO, reconstruction metrics, and KL."""
    data.set_seed()
    train_loader, val_loader, test_loader = data.get_dataloaders()

    model = VAE().to(configs.DEVICE)
    engine.fit(
        model, MODEL_NAME, train_loader, val_loader,
        vae_loss, configs.VAE_LR, configs.VAE_EPOCHS,
    )

    model.eval()
    # Reconstructions use the posterior mean, so the metrics carry no sampling noise.
    test_metrics = metrics.reconstruction_metrics(model.reconstruct, test_loader)
    test_neg_elbo = engine.evaluate_loss(model, test_loader, vae_loss)
    posterior = posterior_statistics(model, test_loader)
    utils.save_json(
        {
            "model": MODEL_NAME,
            "latent_dim": configs.VAE_LATENT_DIM,
            "kl_weight": configs.VAE_KL_WEIGHT,
            "test_neg_elbo": test_neg_elbo,
            **test_metrics,
            **posterior,
        },
        configs.RESULT_DIR / f"{MODEL_NAME}.json",
    )
    print(
        f"test -elbo {test_neg_elbo:.2f} | mse {test_metrics['mse']:.5f} | "
        f"psnr {test_metrics['psnr']:.2f} dB | ssim {test_metrics['ssim']:.4f} | "
        f"kl {posterior['mean_kl_per_image']:.2f} | active dims {posterior['active_latent_dims']}"
    )


if __name__ == "__main__":
    main()
