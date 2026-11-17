"""Step 4: frozen latent decoding of the MLP classifier features.

Freezes the trained MLP classifier (requires_grad=False), trains a separate decoder
64 -> 256 -> 784 with MSE to reconstruct images from the frozen latent z_MLP, and compares
its test MSE/PSNR/SSIM with the Vanilla Autoencoder of the same latent size. The decoder,
optimizer, and number of epochs are identical, so only the encoder differs.
Requires the checkpoints from ``python -m train.train_mlp`` and ``python -m train.train_ae``.

Run from the repository root: ``python -m experiments.mlp_frozen_decoding``.
"""

import matplotlib

matplotlib.use("Agg")  # the script only writes files, so no GUI backend is needed

import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from torch import nn  # noqa: E402

import configs  # noqa: E402
import data  # noqa: E402
import metrics  # noqa: E402
import utils  # noqa: E402
from experiments.mlp_latent_projection import first_image_per_class  # noqa: E402
from models.mlp_classifier import MLPClassifier  # noqa: E402
from models.vanilla_ae import MLPDecoder, VanillaAutoencoder  # noqa: E402
from train import engine  # noqa: E402
from train.train_ae import reconstruction_loss  # noqa: E402

EXPERIMENT_NAME = "mlp_frozen_decoding"
MODEL_NAME = "frozen_latent_decoder"


class FrozenLatentDecoder(nn.Module):
    """A frozen MLP classifier encoder followed by a trainable decoder.

    Only the decoder receives gradients: the classifier weights have
    ``requires_grad=False`` and its forward pass runs under ``torch.no_grad()``.
    """

    def __init__(self, mlp: MLPClassifier) -> None:
        """Freeze the classifier and attach a fresh decoder.

        Args:
            mlp: Trained MLP classifier whose ``encode()`` provides z_MLP.
        """
        super().__init__()
        self.mlp = mlp
        for param in self.mlp.parameters():
            param.requires_grad_(False)
        self.mlp.eval()
        self.decoder = MLPDecoder()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct images from the frozen latent z_MLP.

        Args:
            x: Image batch of shape (B, 1, 28, 28).

        Returns:
            Reconstructions of shape (B, 1, 28, 28).
        """
        with torch.no_grad():
            z_mlp = self.mlp.encode(x)
        return self.decoder(z_mlp)


def plot_comparison(
    originals: torch.Tensor, ae_recon: torch.Tensor, frozen_recon: torch.Tensor
) -> Figure:
    """Plot originals above the Vanilla AE and frozen-decoder reconstructions.

    Args:
        originals: Original images, shape (num_classes, 1, 28, 28).
        ae_recon: Vanilla AE reconstructions, same shape.
        frozen_recon: Frozen-latent decoder reconstructions, same shape.

    Returns:
        A figure with three rows, one column per class.
    """
    rows = (
        ("original", originals),
        (r"Vanilla AE ($z_{AE}$)", ae_recon),
        (r"Frozen decoder ($z_{MLP}$)", frozen_recon),
    )
    fig, axes = plt.subplots(len(rows), configs.NUM_CLASSES, figsize=(15, 5.4))
    for r, (label, images) in enumerate(rows):
        for c in range(configs.NUM_CLASSES):
            ax = axes[r, c]
            ax.imshow(images[c, 0].cpu(), cmap="gray", vmin=0.0, vmax=1.0)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(configs.CLASS_NAMES[c], fontsize=8)
            if c == 0:
                ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=9)
    fig.suptitle("Reconstruction from a 64-d latent: autoencoder vs frozen classifier features")
    fig.tight_layout()
    return fig


def main() -> None:
    """Train the frozen-latent decoder and compare it with the Vanilla AE on the test set."""
    data.set_seed()
    train_loader, val_loader, test_loader = data.get_dataloaders()

    mlp = utils.load_checkpoint(MLPClassifier(), "mlp_classifier")
    frozen = FrozenLatentDecoder(mlp).to(configs.DEVICE)
    engine.fit(
        frozen, MODEL_NAME, train_loader, val_loader,
        reconstruction_loss, configs.AE_LR, configs.AE_EPOCHS,
    )
    frozen.eval()
    ae = utils.load_checkpoint(VanillaAutoencoder(), "vanilla_ae")

    frozen_metrics = metrics.reconstruction_metrics(frozen, test_loader)
    ae_metrics = metrics.reconstruction_metrics(ae, test_loader)
    results = {
        "experiment": EXPERIMENT_NAME,
        "latent_dim": configs.AE_LATENT_DIM,
        "frozen_mlp_decoder": frozen_metrics,
        "vanilla_ae": ae_metrics,
        "mse_ratio_frozen_over_ae": frozen_metrics["mse"] / ae_metrics["mse"],
    }

    originals = first_image_per_class(test_loader).to(configs.DEVICE)
    with torch.no_grad():
        fig = plot_comparison(originals, ae(originals), frozen(originals))
    figure_path = utils.save_figure(fig, EXPERIMENT_NAME)
    plt.close(fig)

    result_path = utils.save_json(results, configs.RESULT_DIR / f"{EXPERIMENT_NAME}.json")
    print(f"figure: {figure_path}")
    print(f"results: {result_path}")


if __name__ == "__main__":
    main()
