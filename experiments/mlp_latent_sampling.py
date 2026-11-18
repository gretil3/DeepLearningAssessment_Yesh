"""Step 5: generative sampling test in the MLP classifier's latent space.

Fits a diagonal Gaussian to the latent features z_MLP of the test set, draws
z_rand ~ N(mu_z, diag(sigma_z^2)), and decodes the draws with the frozen-latent decoder
of Step 4. The decoded images are scored by how close they are to real images, and the
latent statistics show why a Gaussian is a poor description of z_MLP.
Requires the checkpoints from ``python -m train.train_mlp`` and
``python -m experiments.mlp_frozen_decoding``.

Run from the repository root: ``python -m experiments.mlp_latent_sampling``.
"""

import matplotlib

matplotlib.use("Agg")  # the script only writes files, so no GUI backend is needed

import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

import configs  # noqa: E402
import data  # noqa: E402
import utils  # noqa: E402
from experiments.mlp_frozen_decoding import MODEL_NAME, FrozenLatentDecoder  # noqa: E402
from models.mlp_classifier import MLPClassifier  # noqa: E402

EXPERIMENT_NAME = "mlp_latent_sampling"


@torch.no_grad()
def encode_test_set(
    mlp: MLPClassifier, loader: DataLoader
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect the flattened test images and their latent features z_MLP.

    Args:
        mlp: Trained MLP classifier.
        loader: Test dataloader.

    Returns:
        A tuple (images, z) of shapes (N, 784) and (N, 64), both on the CPU.
    """
    images, latents = [], []
    for batch, _ in loader:
        images.append(batch.flatten(start_dim=1))
        latents.append(mlp.encode(batch.to(configs.DEVICE)).cpu())
    return torch.cat(images), torch.cat(latents)


def fit_gaussian(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate the per-dimension mean and standard deviation of the latent features.

    Args:
        z: Latent features of shape (N, 64).

    Returns:
        A tuple (mu_z, sigma_z), each of shape (64,).
    """
    # mu_z = (1/N) sum_i z_i ;  sigma_z^2 = (1/(N-1)) sum_i (z_i - mu_z)^2, per dimension
    return z.mean(dim=0), z.std(dim=0)


def sample_gaussian(
    mu: torch.Tensor, sigma: torch.Tensor, num_samples: int, generator: torch.Generator
) -> torch.Tensor:
    """Draw samples z_rand ~ N(mu_z, diag(sigma_z^2)).

    Args:
        mu: Mean vector of shape (64,).
        sigma: Standard deviation vector of shape (64,).
        num_samples: Number of samples to draw.
        generator: Random generator, so the draw is reproducible.

    Returns:
        Samples of shape (num_samples, 64).
    """
    # Reparameterised draw: z = mu + sigma * eps with eps ~ N(0, I)
    eps = torch.randn(num_samples, mu.numel(), generator=generator)
    return mu + sigma * eps


def nearest_real_mse(
    images: torch.Tensor, reference: torch.Tensor, exclude_self: bool = False
) -> float:
    """Average pixel MSE between each image and its nearest real image.

    A small value means the image looks like some real image; a large value means it
    lies far from the data. This is a rough realism score, not a trained metric.

    Args:
        images: Flattened images of shape (n, 784).
        reference: Flattened real images of shape (N, 784).
        exclude_self: Set True when ``images`` are the first n rows of ``reference``, so an
            image is not matched with itself.

    Returns:
        Mean over ``images`` of the smallest MSE to a real image.
    """
    dist = torch.cdist(images, reference).pow(2) / reference.size(1)
    if exclude_self:
        idx = torch.arange(images.size(0))
        dist[idx, idx] = float("inf")
    return dist.min(dim=1).values.mean().item()


def plot_samples(panels: tuple[tuple[str, torch.Tensor], ...]) -> Figure:
    """Plot several image grids side by side.

    Args:
        panels: Pairs of (title, images) with images of shape (>=16, 1, 28, 28).

    Returns:
        A figure with one grid per panel.
    """
    side = configs.SAMPLING_GRID_SIDE
    fig = plt.figure(figsize=(4.4 * len(panels), 4.8))  # near-square panels, no blank margins
    subfigs = fig.subfigures(1, len(panels))
    for subfig, (title, images) in zip(subfigs, panels):
        subfig.suptitle(title, fontsize=10)
        axes = subfig.subplots(
            side, side,
            gridspec_kw={"left": 0.02, "right": 0.98, "top": 0.88, "bottom": 0.02,
                         "wspace": 0.05, "hspace": 0.05},
        )
        for i, ax in enumerate(axes.flat):
            ax.imshow(images[i, 0], cmap="gray", vmin=0.0, vmax=1.0)
            ax.axis("off")
    return fig


def main() -> None:
    """Sample the fitted Gaussian, decode it, and save the figure and statistics."""
    data.set_seed()
    _, _, test_loader = data.get_dataloaders()
    frozen = utils.load_checkpoint(FrozenLatentDecoder(MLPClassifier()), MODEL_NAME)

    real_images, z_real = encode_test_set(frozen.mlp, test_loader)
    mu, sigma = fit_gaussian(z_real)
    generator = torch.Generator().manual_seed(configs.SEED)
    z_rand = sample_gaussian(mu, sigma, configs.SAMPLING_NUM_EVAL, generator)
    # Ablation: z_MLP is a ReLU output (never negative); clamping removes that mismatch.
    z_clamped = z_rand.clamp(min=0)

    n = configs.SAMPLING_NUM_EVAL
    with torch.no_grad():
        decode = lambda z: frozen.decoder(z.to(configs.DEVICE)).cpu()  # noqa: E731
        x_gauss, x_clamped, x_real = decode(z_rand), decode(z_clamped), decode(z_real[:n])

    results = {
        "experiment": EXPERIMENT_NAME,
        "num_test_images": z_real.size(0),
        "latent_dim": z_real.size(1),
        "num_evaluated_samples": n,
        "latent_gaussian": {
            "mean_of_mu": mu.mean().item(),
            "mean_of_sigma": sigma.mean().item(),
            "dead_dims": int((sigma == 0).sum().item()),
        },
        "real_z_zero_fraction": (z_real == 0).float().mean().item(),
        "gaussian_sample_negative_fraction": (z_rand < 0).float().mean().item(),
        "nearest_real_mse": {
            "real_images_vs_other_real_images": nearest_real_mse(
                real_images[:n], real_images, exclude_self=True
            ),
            "gaussian_samples": nearest_real_mse(x_gauss.flatten(1), real_images),
            "clamped_gaussian_samples": nearest_real_mse(x_clamped.flatten(1), real_images),
        },
    }

    fig = plot_samples(
        (
            (r"Samples $z\sim\mathcal{N}(\mu_z,\ \mathrm{diag}(\sigma_z^2))$", x_gauss),
            (r"Same samples, clamped at 0 (ablation)", x_clamped),
            (r"Decoded real $z_{MLP}$ (reference)", x_real),
        )
    )
    figure_path = utils.save_figure(fig, EXPERIMENT_NAME)
    plt.close(fig)

    result_path = utils.save_json(results, configs.RESULT_DIR / f"{EXPERIMENT_NAME}.json")
    print(f"figure: {figure_path}")
    print(f"results: {result_path}")


if __name__ == "__main__":
    main()
