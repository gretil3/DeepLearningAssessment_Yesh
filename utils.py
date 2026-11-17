"""Shared helpers: wall-clock timing, JSON I/O, training logs, and checkpoints."""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

import configs


class Timer:
    """Context manager that measures wall-clock seconds.

    Attributes:
        elapsed: Seconds spent inside the ``with`` block (set on exit).
    """

    def __init__(self) -> None:
        """Initialise with zero elapsed time."""
        self.elapsed: float = 0.0
        self._start: float = 0.0

    @staticmethod
    def _sync() -> None:
        """Wait for queued GPU work; CUDA kernels are async and would skew timings."""
        if configs.DEVICE.type == "cuda":
            torch.cuda.synchronize()

    def __enter__(self) -> "Timer":
        """Start the clock."""
        self._sync()
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_: object) -> None:
        """Stop the clock and store the elapsed seconds."""
        self._sync()
        self.elapsed = time.perf_counter() - self._start


def save_json(data: Any, path: Path) -> Path:
    """Write ``data`` to ``path`` as indented JSON, creating parent folders.

    Args:
        data: Any JSON-serialisable object.
        path: Destination file.

    Returns:
        The path that was written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return path


@dataclass
class TrainingHistory:
    """Per-epoch losses and wall-clock times of one training run.

    Attributes:
        name: Run identifier, used as the log file name (e.g. ``"vae"``).
        epoch_losses: One dict per epoch, e.g. ``{"train": 0.31, "val": 0.29}``.
        epoch_seconds: Training time of each epoch in seconds.
    """

    name: str
    epoch_losses: list[dict[str, float]] = field(default_factory=list)
    epoch_seconds: list[float] = field(default_factory=list)

    def record(self, losses: dict[str, float], seconds: float) -> None:
        """Append the results of one finished epoch.

        Args:
            losses: Named loss values of the epoch (a dict so GANs can log G and D).
            seconds: Wall-clock time the epoch took.
        """
        self.epoch_losses.append(losses)
        self.epoch_seconds.append(seconds)

    def save(self) -> Path:
        """Write the history to ``outputs/logs/<name>.json``.

        Returns:
            The path of the JSON file.
        """
        payload = {
            "model": self.name,
            "num_epochs": len(self.epoch_seconds),
            "mean_epoch_seconds": sum(self.epoch_seconds) / max(len(self.epoch_seconds), 1),
            "epoch_losses": self.epoch_losses,
            "epoch_seconds": self.epoch_seconds,
        }
        return save_json(payload, configs.LOG_DIR / f"{self.name}.json")


def save_checkpoint(model: nn.Module, name: str) -> Path:
    """Save the model weights to ``outputs/checkpoints/<name>.pt``.

    Args:
        model: Model whose ``state_dict`` is stored.
        name: Checkpoint identifier (e.g. ``"vae"``).

    Returns:
        The path of the checkpoint file.
    """
    path = configs.CHECKPOINT_DIR / f"{name}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    return path


def load_checkpoint(model: nn.Module, name: str) -> nn.Module:
    """Load saved weights into ``model`` and put it in eval mode.

    Args:
        model: Freshly built model with the same architecture as the saved one.
        name: Checkpoint identifier used in :func:`save_checkpoint`.

    Returns:
        The same model, on ``configs.DEVICE`` and in eval mode.
    """
    path = configs.CHECKPOINT_DIR / f"{name}.pt"
    # weights_only=True refuses arbitrary pickled objects; only tensors are needed.
    state = torch.load(path, map_location=configs.DEVICE, weights_only=True)
    model.load_state_dict(state)
    return model.to(configs.DEVICE).eval()
