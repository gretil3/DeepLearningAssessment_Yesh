"""Effect of the L1 sparsity weight lambda on the Sparse Autoencoder (question Q1).

Trains the Sparse Autoencoder once per value of ``configs.SPARSE_LAMBDA_SWEEP`` with the
same data order, initialisation, optimizer, and epochs, then reports test reconstruction
quality (MSE/PSNR/SSIM) and code sparsity for each. lambda = 0 removes the penalty, so it
shows what the same 256-unit ReLU code achieves without any sparsity pressure.

Run from the repository root: ``python -m experiments.sparse_lambda_sweep``.
"""

import csv
from functools import partial
from pathlib import Path
from typing import Any

import configs
import data
import metrics
import utils
from models.sparse_ae import SparseAutoencoder
from train import engine
from train.train_sparse_ae import code_statistics, sparse_loss

EXPERIMENT_NAME = "sparse_lambda_sweep"
CSV_COLUMNS = (
    "lambda", "test_mse", "test_psnr_db", "test_ssim", "mean_l1_norm", "zero_fraction", "mean_active_units",
)


def run_name(lam: float) -> str:
    """Return the log and checkpoint name of one sweep run, e.g. ``sparse_ae_lambda_0.0001``."""
    return f"sparse_ae_lambda_{lam:g}"


def train_and_score(lam: float) -> dict[str, Any]:
    """Train one Sparse Autoencoder and score it on the test set.

    Seeds and dataloaders are rebuilt for every run, so each run starts from the same
    weights and sees the same batches; only lambda differs.

    Args:
        lam: Weight of the L1 penalty.

    Returns:
        Dict with lambda, the test reconstruction metrics, and the sparsity statistics.
    """
    data.set_seed()
    train_loader, val_loader, test_loader = data.get_dataloaders()
    model = SparseAutoencoder().to(configs.DEVICE)
    engine.fit(
        model, run_name(lam), train_loader, val_loader,
        partial(sparse_loss, lam=lam), configs.SPARSE_LR, configs.SPARSE_EPOCHS,
    )
    model.eval()
    return {
        "lambda": lam,
        **metrics.reconstruction_metrics(model, test_loader),
        **code_statistics(model, test_loader),
    }


def save_csv(rows: list[dict[str, Any]], path: Path) -> Path:
    """Write the sweep table as CSV.

    Args:
        rows: One dict per lambda value.
        path: Destination file.

    Returns:
        The path that was written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "lambda": row["lambda"],
                    "test_mse": round(row["mse"], 6),
                    "test_psnr_db": round(row["psnr"], 2),
                    "test_ssim": round(row["ssim"], 4),
                    "mean_l1_norm": round(row["mean_l1_norm"], 3),
                    "zero_fraction": round(row["zero_fraction"], 4),
                    "mean_active_units": round(row["mean_active_units"], 1),
                }
            )
    return path


def main() -> None:
    """Run the sweep and save the comparison table."""
    rows = []
    for lam in configs.SPARSE_LAMBDA_SWEEP:
        print(f"lambda = {lam:g}")
        row = train_and_score(lam)
        rows.append(row)
        print(
            f"  test mse {row['mse']:.5f} | psnr {row['psnr']:.2f} dB | ssim {row['ssim']:.4f} | "
            f"zero units {row['zero_fraction']:.1%} | active units {row['mean_active_units']:.1f}"
        )

    results = {
        "experiment": EXPERIMENT_NAME,
        "code_dim": configs.SPARSE_HIDDEN_DIM,
        "epochs": configs.SPARSE_EPOCHS,
        "runs": rows,
    }
    json_path = utils.save_json(results, configs.RESULT_DIR / f"{EXPERIMENT_NAME}.json")
    csv_path = save_csv(rows, configs.RESULT_DIR / f"{EXPERIMENT_NAME}.csv")
    print(f"results: {json_path}")
    print(f"table: {csv_path}")


if __name__ == "__main__":
    main()
