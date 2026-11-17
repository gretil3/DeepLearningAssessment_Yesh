"""Step 3: latent transpose projection of real test images.

Takes the 64-d latent z_MLP of real test images and maps it straight back to pixel space
with the transposed feature-extractor weights, with no extra training. The output is
compared to the original image to show that the forward classification weights cannot
be inverted into a reconstructive decoder.
Requires the checkpoint from ``python -m train.train_mlp``.

Run from the repository root: ``python -m experiments.mlp_latent_projection``.
"""

from collections.abc import Callable
from functools import partial

import matplotlib

matplotlib.use("Agg")  # the script only writes files, so no GUI backend is needed

import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

import configs  # noqa: E402
import data  # noqa: E402
import metrics  # noqa: E402
import utils  # noqa: E402
from models.mlp_classifier import MLPClassifier  # noqa: E402

EXPERIMENT_NAME = "mlp_latent_projection"


@torch.no_grad()
def project_latent(model: MLPClassifier, z: torch.Tensor) -> torch.Tensor:
    """Map latent features to pixel space with the transposed extractor weights.

    Computes x_hat = Sigmoid(W1^T ReLU(W2^T z_MLP)) with the trained weights.

    Args:
        model: Trained MLP classifier.
        z: Latent features z_MLP of shape (B, 64).

    Returns:
        Back-projected images of shape (B, 784), in [0, 1].
    """
    # nn.Linear stores W as (out, in), the same shape as the assignment's W^(i), so the
    # column-vector product W^T v equals the row-vector product v @ W.
    return torch.sigmoid(torch.relu(z @ model.fc2.weight) @ model.fc1.weight)


@torch.no_grad()
def transpose_reconstruct(model: MLPClassifier, images: torch.Tensor) -> torch.Tensor:
    """Encode images with the MLP, then back-project the latent with W^T.

    Args:
        model: Trained MLP classifier.
        images: Image batch of shape (B, 1, 28, 28).

    Returns:
        Back-projected images of shape (B, 784).
    """
    return project_latent(model, model.encode(images))


def mean_image_reconstructor(loader: DataLoader) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build a reference reconstructor that ignores its input.

    It always returns the average test image, the best constant guess, so any method
    that scores worse than it has not learned to reconstruct anything.

    Args:
        loader: Dataloader to average over (the test loader).

    Returns:
        A function mapping an image batch to copies of the mean image.
    """
    total = torch.zeros(1, configs.IMAGE_SIZE, configs.IMAGE_SIZE)
    count = 0
    for images, _ in loader:
        total += images.sum(dim=0)
        count += images.size(0)
    mean_image = (total / count).to(configs.DEVICE)

    def reconstruct(images: torch.Tensor) -> torch.Tensor:
        """Return the mean image for every item of the batch."""
        return mean_image.expand_as(images)

    return reconstruct


def first_image_per_class(loader: DataLoader) -> torch.Tensor:
    """Collect the first image of every class, in class order.

    Args:
        loader: Unshuffled dataloader (the test loader), so the choice is deterministic.

    Returns:
        Tensor of shape (num_classes, 1, 28, 28) on the CPU.
    """
    found: dict[int, torch.Tensor] = {}
    for images, labels in loader:
        for image, label in zip(images, labels.tolist()):
            found.setdefault(label, image)
        if len(found) == configs.NUM_CLASSES:
            break
    return torch.stack([found[c] for c in range(configs.NUM_CLASSES)])


def plot_comparison(originals: torch.Tensor, projected: torch.Tensor) -> Figure:
    """Plot original test images above their transpose back-projections.

    Args:
        originals: Original images, shape (num_classes, 1, 28, 28).
        projected: Back-projections, shape (num_classes, 784).

    Returns:
        A figure with two rows: the original images and their projections.
    """
    size = configs.IMAGE_SIZE
    rows = (
        ("original", originals.reshape(-1, size, size)),
        (r"$\hat{x}$ (projection)", projected.reshape(-1, size, size)),
    )
    fig, axes = plt.subplots(len(rows), configs.NUM_CLASSES, figsize=(15, 3.8))
    for r, (label, images) in enumerate(rows):
        for c in range(configs.NUM_CLASSES):
            ax = axes[r, c]
            # Fixed 0-1 range keeps both rows on the same scale; no contrast stretching.
            ax.imshow(images[c].cpu(), cmap="gray", vmin=0.0, vmax=1.0)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(configs.CLASS_NAMES[c], fontsize=8)
            if c == 0:
                ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=9)
    fig.suptitle(r"Latent transpose projection of test images: $\hat{x}$ = Sigmoid($W_1^T$ ReLU($W_2^T z_{MLP}$))")
    fig.tight_layout()
    return fig


def main() -> None:
    """Score the projection on the whole test set, save the figure and the results."""
    data.set_seed()
    _, _, test_loader = data.get_dataloaders()
    model = utils.load_checkpoint(MLPClassifier(), "mlp_classifier")

    results = {
        "experiment": EXPERIMENT_NAME,
        "num_test_images": len(test_loader.dataset),
        "latent_transpose_projection": metrics.reconstruction_metrics(
            partial(transpose_reconstruct, model), test_loader
        ),
        "test_mean_image_reference": metrics.reconstruction_metrics(
            mean_image_reconstructor(test_loader), test_loader
        ),
    }

    originals = first_image_per_class(test_loader)
    fig = plot_comparison(originals, transpose_reconstruct(model, originals.to(configs.DEVICE)))
    figure_path = utils.save_figure(fig, EXPERIMENT_NAME)
    plt.close(fig)

    result_path = utils.save_json(results, configs.RESULT_DIR / f"{EXPERIMENT_NAME}.json")
    print(f"figure: {figure_path}")
    print(f"results: {result_path}")


if __name__ == "__main__":
    main()
