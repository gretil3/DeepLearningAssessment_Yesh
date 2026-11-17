"""Shared training loop for models trained with a per-batch loss function."""

import copy
from collections.abc import Callable

import torch
from torch import nn
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader

import configs
import utils

# A loss function takes (model, image batch) and returns the batch-mean scalar loss.
LossFn = Callable[[nn.Module, torch.Tensor], torch.Tensor]


def train_one_epoch(
    model: nn.Module, loader: DataLoader, optimizer: Optimizer, loss_fn: LossFn
) -> float:
    """Run one optimisation pass over the training set.

    Args:
        model: Model to train (switched to train mode).
        loader: Training dataloader.
        optimizer: Optimizer updating the trainable parameters.
        loss_fn: Per-batch loss function.

    Returns:
        Mean loss over the epoch.
    """
    model.train()
    total_loss, total_items = 0.0, 0
    for images, _ in loader:
        images = images.to(configs.DEVICE)
        optimizer.zero_grad()
        loss = loss_fn(model, images)
        loss.backward()
        optimizer.step()
        # Weight by batch size because the last batch is smaller.
        total_loss += loss.item() * images.size(0)
        total_items += images.size(0)
    return total_loss / total_items


@torch.no_grad()
def evaluate_loss(model: nn.Module, loader: DataLoader, loss_fn: LossFn) -> float:
    """Compute the mean loss on a dataloader.

    Args:
        model: Model to evaluate (switched to eval mode).
        loader: Dataloader to score.
        loss_fn: Per-batch loss function.

    Returns:
        Mean loss over the loader.
    """
    model.eval()
    total_loss, total_items = 0.0, 0
    for images, _ in loader:
        images = images.to(configs.DEVICE)
        total_loss += loss_fn(model, images).item() * images.size(0)
        total_items += images.size(0)
    return total_loss / total_items


def fit(
    model: nn.Module,
    name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: LossFn,
    lr: float,
    epochs: int,
) -> utils.TrainingHistory:
    """Train a model, keep the best epoch by validation loss, and save the artifacts.

    Writes the checkpoint to ``outputs/checkpoints/<name>.pt`` and the per-epoch loss and
    training time to ``outputs/logs/<name>.json``. Parameters with ``requires_grad=False``
    (for example a frozen encoder) are left out of the optimizer.

    Args:
        model: Model already moved to ``configs.DEVICE``.
        name: Run identifier used for the checkpoint and log file names.
        train_loader: Training dataloader.
        val_loader: Validation dataloader, used only to pick the epoch.
        loss_fn: Per-batch loss function.
        lr: Adam learning rate.
        epochs: Number of epochs.

    Returns:
        The recorded training history. The model holds the best weights on return.
    """
    optimizer = Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    history = utils.TrainingHistory(name)

    # The epoch is chosen on the validation set so the test set stays untouched.
    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(1, epochs + 1):
        with utils.Timer() as timer:
            train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn)
        val_loss = evaluate_loss(model, val_loader, loss_fn)
        history.record({"train_loss": train_loss, "val_loss": val_loss}, timer.elapsed)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
        print(
            f"epoch {epoch:02d}/{epochs} | train loss {train_loss:.5f} | "
            f"val loss {val_loss:.5f} | {timer.elapsed:.1f}s"
        )

    model.load_state_dict(best_state)
    utils.save_checkpoint(model, name)
    history.save()
    return history
