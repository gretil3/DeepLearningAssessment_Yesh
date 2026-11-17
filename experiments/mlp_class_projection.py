"""Step 2: top-down weight transposition from one-hot class vectors to pixel space.

Back-projects each class vector through the transposed, trained MLP weights with no
extra training, to show that W^T is not W^-1 and cannot act as an image decoder.
Requires the checkpoint from ``python -m train.train_mlp``.

Run from the repository root: ``python -m experiments.mlp_class_projection``.
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
from models.mlp_classifier import MLPClassifier  # noqa: E402

EXPERIMENT_NAME = "mlp_class_projection"


@torch.no_grad()
def project_class_vectors(model: MLPClassifier) -> torch.Tensor:
    """Back-project every one-hot class vector to pixel space with transposed weights.

    Computes, with the trained weights and no further training:
        z_hat  = ReLU(W3^T y_c)
        h1_hat = ReLU(W2^T z_hat)
        x_hat  = Sigmoid(W1^T h1_hat)

    Args:
        model: Trained MLP classifier.

    Returns:
        Tensor of shape (num_classes, 784), one back-projected image per class.
    """
    # nn.Linear stores W as (out, in), the same shape as the assignment's W^(i), so the
    # column-vector product W^T v equals the row-vector product v @ W.
    one_hot = torch.eye(configs.NUM_CLASSES, device=configs.DEVICE)
    z_hat = torch.relu(one_hot @ model.fc3.weight)
    h1_hat = torch.relu(z_hat @ model.fc2.weight)
    return torch.sigmoid(h1_hat @ model.fc1.weight)


def class_mean_images(loader: DataLoader) -> torch.Tensor:
    """Average the flattened images of each class, as a reference template.

    Args:
        loader: Dataloader to average over (the test loader).

    Returns:
        Tensor of shape (num_classes, 784) on the CPU.
    """
    sums = torch.zeros(configs.NUM_CLASSES, configs.INPUT_DIM)
    counts = torch.zeros(configs.NUM_CLASSES)
    for images, labels in loader:
        sums.index_add_(0, labels, images.flatten(start_dim=1))
        counts.index_add_(0, labels, torch.ones(len(labels)))
    return sums / counts.unsqueeze(1)


@torch.no_grad()
def weight_orthogonality(model: MLPClassifier) -> dict[str, dict[str, float]]:
    """Measure how far each weight matrix is from being orthogonal.

    W^T would equal W^-1 only if W were orthogonal. For each layer this reports
    ||W W^T - I||_F / ||I||_F and ||W^T W - I||_F / ||I||_F. A rectangular W (out < in)
    has rank <= out, so W^T W can never be the identity.

    Args:
        model: Trained MLP classifier.

    Returns:
        Per-layer dict with the matrix shape, its rank, and both deviations.
    """
    report: dict[str, dict[str, float]] = {}
    for name, layer in (("W1", model.fc1), ("W2", model.fc2), ("W3", model.fc3)):
        w = layer.weight
        out_dim, in_dim = w.shape
        dev_wwt = torch.linalg.norm(w @ w.T - torch.eye(out_dim, device=w.device)) / out_dim**0.5
        dev_wtw = torch.linalg.norm(w.T @ w - torch.eye(in_dim, device=w.device)) / in_dim**0.5
        report[name] = {
            "shape": [out_dim, in_dim],
            "rank": int(torch.linalg.matrix_rank(w).item()),
            "rel_dev_wwt_from_identity": dev_wwt.item(),
            "rel_dev_wtw_from_identity": dev_wtw.item(),
        }
    return report


def plot_projection(x_hat: torch.Tensor, class_means: torch.Tensor) -> Figure:
    """Plot the back-projections next to the class-mean reference images.

    Args:
        x_hat: Back-projected images, shape (num_classes, 784).
        class_means: Class-mean test images, shape (num_classes, 784).

    Returns:
        A figure with three rows: raw x_hat, contrast-stretched x_hat, class means.
    """
    rows = (
        (r"$\hat{x}$ (raw, 0-1)", x_hat, (0.0, 1.0)),
        (r"$\hat{x}$ (stretched)", x_hat, (None, None)),  # per-image min-max to reveal structure
        ("class mean", class_means, (0.0, 1.0)),
    )
    fig, axes = plt.subplots(len(rows), configs.NUM_CLASSES, figsize=(15, 5.4))
    for r, (label, images, (vmin, vmax)) in enumerate(rows):
        for c in range(configs.NUM_CLASSES):
            ax = axes[r, c]
            ax.imshow(
                images[c].cpu().reshape(configs.IMAGE_SIZE, configs.IMAGE_SIZE),
                cmap="gray", vmin=vmin, vmax=vmax,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(configs.CLASS_NAMES[c], fontsize=8)
            if c == 0:
                ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=9)
    fig.suptitle("Top-down weight transposition of one-hot class vectors (no training)")
    fig.tight_layout()
    return fig


def main() -> None:
    """Run the projection, save the figure, and write the quantitative results."""
    data.set_seed()
    _, _, test_loader = data.get_dataloaders()
    model = utils.load_checkpoint(MLPClassifier(), "mlp_classifier")

    x_hat = project_class_vectors(model)
    class_means = class_mean_images(test_loader)

    with torch.no_grad():
        predictions = model(x_hat).argmax(dim=1).cpu()

    per_class = []
    for c in range(configs.NUM_CLASSES):
        corr = torch.corrcoef(torch.stack([x_hat[c].cpu(), class_means[c]]))[0, 1].item()
        per_class.append(
            {
                "class": configs.CLASS_NAMES[c],
                "corr_with_class_mean": corr,
                "classifier_prediction": configs.CLASS_NAMES[int(predictions[c])],
            }
        )

    fig = plot_projection(x_hat, class_means)
    figure_path = utils.save_figure(fig, EXPERIMENT_NAME)
    plt.close(fig)

    results = {
        "experiment": EXPERIMENT_NAME,
        "pixel_min": x_hat.min().item(),
        "pixel_max": x_hat.max().item(),
        "pixel_mean": x_hat.mean().item(),
        "mean_corr_with_class_mean": sum(p["corr_with_class_mean"] for p in per_class) / len(per_class),
        "classifier_agrees_with_class": int((predictions == torch.arange(configs.NUM_CLASSES)).sum()),
        "per_class": per_class,
        "weight_orthogonality": weight_orthogonality(model),
    }
    result_path = utils.save_json(results, configs.RESULT_DIR / f"{EXPERIMENT_NAME}.json")
    print(f"figure: {figure_path}")
    print(f"results: {result_path}")


if __name__ == "__main__":
    main()
