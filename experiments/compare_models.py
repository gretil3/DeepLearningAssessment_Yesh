"""Quantitative comparison of the six generative models (Part IV table of the report).

Collects, for every model, the training time per epoch and loss-stability statistics
(from the training logs), the test reconstruction metrics of the autoencoders (from their
result files), the inference latency per batch of 64 samples (measured here), and the
FID of newly generated images against the test set (computed here).
Requires every training script to have been run first.

Run from the repository root: ``python -m experiments.compare_models``.
The DDPM needs 1000 network calls per image, so its 1000 FID samples take a long time on
a CPU. Generated images are cached in ``outputs/samples/`` and the run resumes where it
stopped if interrupted.
"""

import csv
import statistics
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

import configs
import data
import utils
from metrics import FidScorer
from models import zoo

EXPERIMENT_NAME = "model_comparison"
NOT_AVAILABLE = "N/A"

# Display name, latent size shown in the table, and what the latency measures.
# Autoencoders are timed on reconstruction; models used as generators on sampling.
MODELS: dict[str, dict[str, str]] = {
    "vanilla_ae": {"title": "Vanilla Autoencoder", "latent": str(configs.AE_LATENT_DIM), "op": "reconstruction"},
    "sparse_ae": {"title": "Sparse Autoencoder", "latent": str(configs.SPARSE_HIDDEN_DIM), "op": "reconstruction"},
    "vae": {"title": "VAE", "latent": str(configs.VAE_LATENT_DIM), "op": "generation"},
    "vanilla_gan": {"title": "Vanilla GAN", "latent": str(configs.GAN_LATENT_DIM), "op": "generation"},
    "dcgan": {"title": "DCGAN", "latent": str(configs.DCGAN_LATENT_DIM), "op": "generation"},
    "ddpm": {"title": "DDPM", "latent": f"{configs.IMAGE_SIZE}x{configs.IMAGE_SIZE} (image space)", "op": "generation"},
}
# Models whose reconstruction metrics exist (the others have no encoder).
RECONSTRUCTION_MODELS = zoo.AUTOENCODER_NAMES
CSV_COLUMNS = (
    "model", "latent_dim", "epochs", "train_seconds_per_epoch", "latency_ms_batch64", "latency_op",
    "test_mse", "test_psnr_db", "test_ssim", "fid",
)


def collect_images(loader: DataLoader) -> torch.Tensor:
    """Gather all images of a loader into one CPU tensor.

    Args:
        loader: Test dataloader.

    Returns:
        Images of shape (N, 1, 28, 28) in [0, 1].
    """
    return torch.cat([images for images, _ in loader])


def generate_samples(
    name: str, sample_fn: Callable[[int], torch.Tensor], num_samples: int
) -> torch.Tensor:
    """Generate images in batches and cache them, resuming from an earlier partial run.

    Each batch is seeded from its index, so a resumed run produces exactly the images an
    uninterrupted run would. The cache is tied to the checkpoint's modification time, so
    retraining the model invalidates it.

    Args:
        name: One of ``zoo.GENERATOR_NAMES``; also the cache file name.
        sample_fn: Function drawing ``n`` images from the model.
        num_samples: Number of images wanted.

    Returns:
        Images of shape (num_samples, 1, 28, 28) on the CPU.
    """
    batch_size = configs.LATENCY_BATCH_SIZE
    cache_path = configs.SAMPLE_DIR / f"{name}.pt"
    stamp = (configs.CHECKPOINT_DIR / f"{zoo.GENERATOR_CHECKPOINTS[name]}.pt").stat().st_mtime_ns

    images = torch.empty(0, 1, configs.IMAGE_SIZE, configs.IMAGE_SIZE)
    if cache_path.exists():
        cached = torch.load(cache_path, weights_only=True)
        if cached["checkpoint_mtime_ns"] == stamp:
            images = cached["images"]

    while images.size(0) < num_samples:
        # Seed from the batch index so resuming reproduces the same stream of batches.
        torch.manual_seed(configs.SEED + images.size(0) // batch_size)
        batch = sample_fn(min(batch_size, num_samples - images.size(0))).cpu()
        images = torch.cat([images, batch])
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_suffix(".tmp")
        torch.save({"images": images, "checkpoint_mtime_ns": stamp}, tmp)
        tmp.replace(cache_path)
        print(f"  {name}: {images.size(0)}/{num_samples} samples", flush=True)
    return images[:num_samples]


@torch.no_grad()
def measure_latency(run: Callable[[], Any], warmup: int, repeats: int) -> dict[str, float]:
    """Time a function that processes one batch.

    Args:
        run: Function that performs one full batch of inference.
        warmup: Untimed calls made first.
        repeats: Timed calls.

    Returns:
        Dict with the mean and standard deviation in milliseconds and the repeat count.
    """
    for _ in range(warmup):
        run()
    milliseconds = []
    for _ in range(repeats):
        with utils.Timer() as timer:
            run()
        milliseconds.append(timer.elapsed * 1000.0)
    return {
        "mean_ms": statistics.fmean(milliseconds),
        "std_ms": statistics.pstdev(milliseconds),
        "repeats": repeats,
    }


def latency_runner(name: str, test_images: torch.Tensor) -> Callable[[], Any]:
    """Build the one-batch inference function whose latency is reported for a model.

    Args:
        name: Key of ``MODELS``.
        test_images: Test images; the first batch is used for reconstruction timing.

    Returns:
        A function that runs one batch of ``configs.LATENCY_BATCH_SIZE`` samples.
    """
    batch_size = configs.LATENCY_BATCH_SIZE
    if MODELS[name]["op"] == "generation":
        sample = zoo.load_prior_sampler(name)
        return lambda: sample(batch_size)
    model = zoo.load_autoencoder(name)
    images = test_images[:batch_size].to(configs.DEVICE)
    return lambda: model.decode(model.encode(images))


def training_summary(name: str) -> dict[str, Any]:
    """Summarise a training log: time per epoch and how settled each loss is.

    For every logged loss the final value, the mean over the last epochs, and their
    relative standard deviation (std / |mean|) are reported. The relative standard
    deviation still contains any remaining downward trend, so it is an upper bound on
    the epoch-to-epoch noise.

    Args:
        name: Name of the log file in ``outputs/logs/`` (without extension).

    Returns:
        Dict with the epoch count, mean seconds per epoch, and per-loss statistics.
    """
    log = utils.load_json(configs.LOG_DIR / f"{name}.json")
    tail = configs.STABILITY_TAIL_EPOCHS
    losses: dict[str, dict[str, float]] = {}
    for key in log["epoch_losses"][0]:
        values = [epoch[key] for epoch in log["epoch_losses"]]
        tail_values = values[-tail:]
        tail_mean = statistics.fmean(tail_values)
        losses[key] = {
            "final": values[-1],
            "tail_mean": tail_mean,
            "tail_relative_std": statistics.pstdev(tail_values) / abs(tail_mean),
        }
    return {
        "epochs": log["num_epochs"],
        "mean_epoch_seconds": log["mean_epoch_seconds"],
        "tail_epochs": tail,
        "losses": losses,
    }


def build_rows(report: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten the report into one row per model, in the layout of the report table.

    Args:
        report: Per-model results gathered by :func:`main`.

    Returns:
        Rows with the keys of ``CSV_COLUMNS``; unavailable values are ``N/A``.
    """
    rows = []
    for name, info in MODELS.items():
        entry = report[name]
        recon = entry.get("reconstruction", {})
        rows.append(
            {
                "model": info["title"],
                "latent_dim": info["latent"],
                "epochs": entry["training"]["epochs"],
                "train_seconds_per_epoch": round(entry["training"]["mean_epoch_seconds"], 2),
                "latency_ms_batch64": round(entry["latency"]["mean_ms"], 2),
                "latency_op": info["op"],
                "test_mse": round(recon["mse"], 6) if recon else NOT_AVAILABLE,
                "test_psnr_db": round(recon["psnr"], 2) if recon else NOT_AVAILABLE,
                "test_ssim": round(recon["ssim"], 4) if recon else NOT_AVAILABLE,
                "fid": round(entry["fid"], 2) if "fid" in entry else NOT_AVAILABLE,
            }
        )
    return rows


def save_csv(rows: list[dict[str, Any]], path: Path) -> Path:
    """Write the table rows as CSV.

    Args:
        rows: Rows from :func:`build_rows`.
        path: Destination file.

    Returns:
        The path that was written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def main() -> None:
    """Measure FID and latency, merge them with the logs, and save the comparison table."""
    data.set_seed()
    _, _, test_loader = data.get_dataloaders()
    test_images = collect_images(test_loader)
    n = configs.FID_NUM_SAMPLES

    report: dict[str, dict[str, Any]] = {}
    for name in MODELS:
        entry: dict[str, Any] = {"training": training_summary(name)}
        if name in RECONSTRUCTION_MODELS:
            entry["reconstruction"] = utils.load_json(configs.RESULT_DIR / f"{name}.json")
        report[name] = entry

    print("FID: extracting real-image features")
    scorer = FidScorer(test_images)
    for name in zoo.GENERATOR_NAMES:
        print(f"FID: sampling {name}")
        samples = generate_samples(name, zoo.load_prior_sampler(name), n)
        report[name]["fid"] = scorer.score(samples)
        print(f"  {name}: FID {report[name]['fid']:.2f}")

    # Reference level: n real test images scored against the remaining real images. It
    # shows the lowest FID that this sample size can reach.
    baseline = FidScorer(test_images[n:]).score(test_images[:n])
    print(f"FID of {n} real images against the other real images: {baseline:.2f}")

    for name, entry in report.items():
        print(f"latency: {name}")
        repeats = configs.LATENCY_REPEATS_DDPM if name == "ddpm" else configs.LATENCY_REPEATS
        warmup = 0 if name == "ddpm" else configs.LATENCY_WARMUP
        entry["latency"] = measure_latency(latency_runner(name, test_images), warmup, repeats)

    rows = build_rows(report)
    results = {
        "experiment": EXPERIMENT_NAME,
        "device": configs.DEVICE.type,
        "latency_batch_size": configs.LATENCY_BATCH_SIZE,
        "fid_num_samples": n,
        "fid_num_real_images": test_images.size(0),
        "fid_real_vs_real_baseline": baseline,
        "models": report,
    }
    json_path = utils.save_json(results, configs.RESULT_DIR / f"{EXPERIMENT_NAME}.json")
    csv_path = save_csv(rows, configs.RESULT_DIR / f"{EXPERIMENT_NAME}.csv")

    for row in rows:
        print(" | ".join(str(row[column]) for column in CSV_COLUMNS))
    print(f"results: {json_path}")
    print(f"table: {csv_path}")


if __name__ == "__main__":
    main()
