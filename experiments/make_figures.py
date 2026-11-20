"""Generate the five comparison figures of the report (Part V).

1. Reconstruction grid: original images against Vanilla AE, Sparse AE, and VAE.
2. Random generative sampling: 4x4 samples of VAE, Vanilla GAN, DCGAN, and DDPM.
3. Evidence of the MLP classifier's limits: transpose projections, frozen decoder, prior sampling.
4. Linear latent interpolation between two images of contrasting classes.
5. Training curves of all six models and the DDPM reverse-denoising trajectory.

Requires all training scripts, ``experiments.mlp_frozen_decoding``, and
``experiments.compare_models`` (whose cached samples are reused for Figure 2).

Run from the repository root: ``python -m experiments.make_figures``.
"""

from collections.abc import Callable

import matplotlib

matplotlib.use("Agg")  # the script only writes files, so no GUI backend is needed

import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

import configs  # noqa: E402
import data  # noqa: E402
import utils  # noqa: E402
from experiments.compare_models import generate_samples  # noqa: E402
from experiments.mlp_class_projection import project_class_vectors  # noqa: E402
from experiments.mlp_frozen_decoding import MODEL_NAME as FROZEN_NAME  # noqa: E402
from experiments.mlp_frozen_decoding import FrozenLatentDecoder  # noqa: E402
from experiments.mlp_latent_projection import first_image_per_class, transpose_reconstruct  # noqa: E402
from experiments.mlp_latent_sampling import (  # noqa: E402
    encode_test_set,
    fit_gaussian,
    plot_samples,
    sample_gaussian,
)
from models import zoo  # noqa: E402
from models.mlp_classifier import MLPClassifier  # noqa: E402

# Figure names: official numbering first, so the files sort in report order.
FIGURE_NAMES = {
    1: "fig1_reconstruction_grid",
    2: "fig2_random_sampling",
    3: "fig3_mlp_limitation_evidence",
    4: "fig4_latent_interpolation",
    5: "fig5_training_curves_and_ddpm_trajectory",
}
SPACER = None  # marks a thin blank row inside ``plot_image_rows``


def show_image(ax: plt.Axes, image: torch.Tensor) -> None:
    """Draw one grey-scale image on a fixed 0-1 scale, without ticks.

    Args:
        ax: Target axes.
        image: Tensor with 784 values (any shape) in [0, 1].
    """
    ax.imshow(image.detach().cpu().reshape(configs.IMAGE_SIZE, configs.IMAGE_SIZE),
              cmap="gray", vmin=0.0, vmax=1.0)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_image_rows(
    rows: list[tuple[str, torch.Tensor] | None],
    column_titles: list[str],
    title: str,
) -> Figure:
    """Plot labelled rows of images that share the same columns.

    Args:
        rows: One (label, images) pair per row, images holding one item per column; a
            ``SPACER`` entry leaves a thin gap between groups of rows.
        column_titles: Title above each column.
        title: Figure title.

    Returns:
        The figure.
    """
    num_cols = len(column_titles)
    ratios = [0.25 if row is SPACER else 1.0 for row in rows]
    fig = plt.figure(figsize=(1.05 * num_cols + 1.8, 1.05 * sum(ratios) + 0.9))
    grid = fig.add_gridspec(len(rows), num_cols, height_ratios=ratios, hspace=0.06, wspace=0.05)
    first_visible = True
    for r, row in enumerate(rows):
        if row is SPACER:
            continue
        label, images = row
        for c in range(num_cols):
            ax = fig.add_subplot(grid[r, c])
            show_image(ax, images[c])
            if first_visible:
                ax.set_title(column_titles[c], fontsize=8)
            if c == 0:
                ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=8)
        first_visible = False
    fig.suptitle(title, fontsize=10)
    return fig


@torch.no_grad()
def figure_reconstruction(test_images: torch.Tensor) -> Figure:
    """Figure 1: originals above the reconstructions of the three autoencoders.

    Args:
        test_images: One test image per class, shape (10, 1, 28, 28), on ``configs.DEVICE``.

    Returns:
        The figure.
    """
    vanilla = zoo.load_autoencoder("vanilla_ae")
    sparse = zoo.load_autoencoder("sparse_ae")
    vae = zoo.load_autoencoder("vae")
    rows = [
        ("Original", test_images),
        ("Vanilla AE", vanilla.decode(vanilla.encode(test_images))),
        ("Sparse AE", sparse.decode(sparse.encode(test_images))),
        ("VAE", vae.reconstruct(test_images)),
    ]
    return plot_image_rows(rows, list(configs.CLASS_NAMES), "Reconstruction of test images")


def figure_sampling() -> Figure:
    """Figure 2: 4x4 random samples of the four generative models.

    Returns:
        The figure, with one panel per model in a 2x2 layout.
    """
    n = configs.SAMPLING_GRID_SIDE**2
    titles = {
        "vae": r"(A) VAE, $z\sim\mathcal{N}(0,I)$",
        "vanilla_gan": r"(B) Vanilla GAN, $z\sim\mathcal{N}(0,I)$",
        "dcgan": r"(C) DCGAN, $z\sim\mathcal{N}(0,I)$",
        "ddpm": r"(D) DDPM, $x_T\sim\mathcal{N}(0,I)$ reverse denoising",
    }
    # These are the first samples of the cached evaluation sets, so the figure shows the
    # same images that the FID scores were computed from.
    panels = tuple(
        (titles[name], generate_samples(name, zoo.load_prior_sampler(name), configs.FID_NUM_SAMPLES)[:n])
        for name in zoo.GENERATOR_NAMES
    )
    return plot_samples(panels, num_rows=2)


@torch.no_grad()
def figure_mlp_evidence(test_images: torch.Tensor) -> Figure:
    """Figure 3: the four pieces of evidence that classifier features cannot generate images.

    Rows: (1) original; (2) transpose projection of the class vector and of z_MLP;
    (3) Vanilla AE against the frozen-latent decoder; (4) samples from N(mu_z, diag(sigma_z^2)).

    Args:
        test_images: One test image per class, shape (10, 1, 28, 28), on ``configs.DEVICE``.

    Returns:
        The figure.
    """
    _, _, test_loader = data.get_dataloaders()
    frozen = utils.load_checkpoint(FrozenLatentDecoder(MLPClassifier()), FROZEN_NAME)
    mlp = frozen.mlp
    ae = zoo.load_autoencoder("vanilla_ae")

    _, z_real = encode_test_set(mlp, test_loader)
    mu, sigma = fit_gaussian(z_real)
    generator = torch.Generator().manual_seed(configs.SEED)  # same draws as the Step 5 script
    z_rand = sample_gaussian(mu, sigma, configs.NUM_CLASSES, generator)

    rows = [
        ("(1) Original", test_images),
        SPACER,
        (r"(2) $W^T$ from class $y_c$", project_class_vectors(mlp)),
        (r"(2) $W^T$ from $z_{MLP}$", transpose_reconstruct(mlp, test_images)),
        SPACER,
        (r"(3) Vanilla AE ($z_{AE}$)", ae(test_images)),
        (r"(3) Frozen decoder ($z_{MLP}$)", frozen(test_images)),
        SPACER,
        (r"(4) $z\sim\mathcal{N}(\mu_z,\sigma_z^2)$", frozen.decoder(z_rand.to(configs.DEVICE))),
    ]
    return plot_image_rows(
        rows, list(configs.CLASS_NAMES),
        "Why an MLP classifier cannot reconstruct or generate images (row 4 is random, not class-aligned)",
    )


@torch.no_grad()
def pick_clear_latents(
    decode: Callable[[torch.Tensor], torch.Tensor], mlp: MLPClassifier, latent_dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Find prior latents that a generator turns into the two chosen classes.

    GANs have no encoder, so their endpoints are random latents. To get two contrasting
    endpoints, ``INTERPOLATION_CANDIDATES`` latents are drawn and the one that the MLP
    classifier is most confident about is kept for each class.

    Args:
        decode: Generator mapping (B, latent_dim) to images.
        mlp: Trained MLP classifier, used only to label the generated images.
        latent_dim: Size of the latent vector.

    Returns:
        A tuple (z_A, z_B), each of shape (1, latent_dim).
    """
    z = torch.randn(configs.INTERPOLATION_CANDIDATES, latent_dim, device=configs.DEVICE)
    probs = mlp(decode(z)).softmax(dim=1)
    class_a, class_b = configs.INTERPOLATION_CLASSES
    return z[probs[:, class_a].argmax()].unsqueeze(0), z[probs[:, class_b].argmax()].unsqueeze(0)


def interpolate(decode: Callable[[torch.Tensor], torch.Tensor], z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
    """Decode the linear path between two latents.

    Args:
        decode: Maps (B, latent_dim) to images.
        z_a: Start latent of shape (1, latent_dim).
        z_b: End latent of shape (1, latent_dim).

    Returns:
        Decoded images, one per value of ``INTERPOLATION_ALPHAS``.
    """
    alphas = torch.tensor(configs.INTERPOLATION_ALPHAS, device=configs.DEVICE).unsqueeze(1)
    # z(alpha) = (1 - alpha) z_A + alpha z_B
    return decode((1.0 - alphas) * z_a + alphas * z_b)


@torch.no_grad()
def figure_interpolation(test_images: torch.Tensor) -> Figure:
    """Figure 4: linear latent interpolation for the five latent-space models.

    The autoencoders interpolate between the codes of two real test images of the chosen
    classes (the VAE between their posterior means); the GANs between two random latents
    that decode to those classes.

    Args:
        test_images: One test image per class, shape (10, 1, 28, 28), on ``configs.DEVICE``.

    Returns:
        The figure.
    """
    class_a, class_b = configs.INTERPOLATION_CLASSES
    pair = test_images[[class_a, class_b]]
    mlp = utils.load_checkpoint(MLPClassifier(), "mlp_classifier")
    rows: list[tuple[str, torch.Tensor] | None] = []

    for name, title in (("vanilla_ae", "Vanilla AE"), ("sparse_ae", "Sparse AE")):
        model = zoo.load_autoencoder(name)
        z = model.encode(pair)
        rows.append((title, interpolate(model.decode, z[:1], z[1:])))
    vae = zoo.load_autoencoder("vae")
    mu, _ = vae.encode(pair)
    rows.append(("VAE", interpolate(vae.decode, mu[:1], mu[1:])))

    torch.manual_seed(configs.SEED)
    for name, title in (("vanilla_gan", "Vanilla GAN"), ("dcgan", "DCGAN")):
        decode, latent_dim = zoo.load_decoder(name)
        z_a, z_b = pick_clear_latents(decode, mlp, latent_dim)
        rows.append((title, interpolate(decode, z_a, z_b)))

    columns = [rf"$\alpha$ = {a:.1f}" for a in configs.INTERPOLATION_ALPHAS]
    return plot_image_rows(
        rows, columns,
        rf"Linear latent interpolation: {configs.CLASS_NAMES[class_a]} ($\alpha$=0) to "
        rf"{configs.CLASS_NAMES[class_b]} ($\alpha$=1)",
    )


def plot_loss_curves(axes: list[plt.Axes]) -> None:
    """Draw the per-epoch loss curves of the six models from their training logs.

    Args:
        axes: Six axes, in the order of ``panels`` below.
    """
    panels = (
        ("vanilla_ae", "Vanilla AE", "MSE", ("train_loss", "val_loss")),
        ("sparse_ae", "Sparse AE", r"MSE + $\lambda\|h\|_1$", ("train_loss", "val_loss")),
        ("vae", "VAE", "Negative ELBO per image", ("train_loss", "val_loss")),
        ("vanilla_gan", "Vanilla GAN", "BCE loss", ("d_loss", "g_loss")),
        ("dcgan", "DCGAN", "BCE loss", ("d_loss", "g_loss")),
        ("ddpm", "DDPM", r"$\|\epsilon-\epsilon_\theta\|^2$ (MSE)", ("train_loss", "val_loss")),
    )
    legend = {"train_loss": "train", "val_loss": "validation", "d_loss": "Discriminator", "g_loss": "Generator"}
    for ax, (name, title, ylabel, keys) in zip(axes, panels):
        epochs = utils.load_json(configs.LOG_DIR / f"{name}.json")["epoch_losses"]
        for key in keys:
            ax.plot(range(1, len(epochs) + 1), [e[key] for e in epochs], label=legend[key])
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("epoch", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)


@torch.no_grad()
def figure_training_and_trajectory() -> Figure:
    """Figure 5: loss curves of the six models above the DDPM reverse trajectory.

    Returns:
        The figure.
    """
    steps = configs.DDPM_TRAJECTORY_STEPS
    torch.manual_seed(configs.SEED)
    _, snapshots = zoo.load_diffusion().sample_trajectory(configs.DDPM_TRAJECTORY_SAMPLES, steps)

    fig = plt.figure(figsize=(12, 11))
    top, bottom = fig.subfigures(2, 1, height_ratios=[2.0, 1.25])
    # Wide gaps keep the axis labels clear of the neighbouring panels' titles and ticks.
    plot_loss_curves(list(top.subplots(2, 3, gridspec_kw={"hspace": 0.55, "wspace": 0.4, "top": 0.9}).flat))
    top.suptitle("Training loss per epoch", fontsize=11)

    bottom.suptitle(r"DDPM reverse denoising trajectory ($x_T \to x_0$)", fontsize=11)
    axes = bottom.subplots(
        configs.DDPM_TRAJECTORY_SAMPLES, len(steps),
        gridspec_kw={"hspace": 0.05, "wspace": 0.05, "top": 0.86, "bottom": 0.02},
    )
    for c, t in enumerate(steps):
        for r in range(configs.DDPM_TRAJECTORY_SAMPLES):
            show_image(axes[r, c], snapshots[t][r])
        axes[0, c].set_title(f"t = {t}", fontsize=9)
    return fig


def main() -> None:
    """Build the five figures and save them as high-resolution PNG files."""
    data.set_seed()
    _, _, test_loader = data.get_dataloaders()
    test_images = first_image_per_class(test_loader).to(configs.DEVICE)

    builders = {
        1: lambda: figure_reconstruction(test_images),
        2: figure_sampling,
        3: lambda: figure_mlp_evidence(test_images),
        4: lambda: figure_interpolation(test_images),
        5: figure_training_and_trajectory,
    }
    for number, build in builders.items():
        fig = build()
        path = utils.save_figure(fig, FIGURE_NAMES[number])
        plt.close(fig)
        print(f"figure {number}: {path}")


if __name__ == "__main__":
    main()
