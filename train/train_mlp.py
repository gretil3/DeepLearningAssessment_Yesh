"""Train the MLP classifier (Step 1 of the weight-transposition experiment).

Run from the repository root: ``python -m train.train_mlp``.
"""

import copy

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

import configs
import data
import utils
from models.mlp_classifier import MLPClassifier

MODEL_NAME = "mlp_classifier"


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: Adam) -> float:
    """Run one optimisation pass over the training set.

    Args:
        model: Classifier to train (switched to train mode).
        loader: Training dataloader.
        optimizer: Optimizer updating the model parameters.

    Returns:
        Mean cross-entropy loss over the epoch.
    """
    model.train()
    total_loss, total_items = 0.0, 0
    for images, labels in loader:
        images, labels = images.to(configs.DEVICE), labels.to(configs.DEVICE)
        optimizer.zero_grad()
        # Cross-entropy on logits z: L = -(1/N) * sum_i log softmax(z_i)[y_i]
        loss = F.cross_entropy(model(images), labels)
        loss.backward()
        optimizer.step()
        # Weight by batch size because the last batch is smaller.
        total_loss += loss.item() * images.size(0)
        total_items += images.size(0)
    return total_loss / total_items


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader) -> tuple[float, float]:
    """Compute mean cross-entropy loss and accuracy on a dataloader.

    Args:
        model: Classifier to evaluate (switched to eval mode).
        loader: Dataloader to score.

    Returns:
        A (loss, accuracy) tuple.
    """
    model.eval()
    total_loss, correct, total_items = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(configs.DEVICE), labels.to(configs.DEVICE)
        logits = model(images)
        total_loss += F.cross_entropy(logits, labels, reduction="sum").item()
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total_items += images.size(0)
    return total_loss / total_items, correct / total_items


def main() -> None:
    """Train, keep the best epoch by validation loss, then report test accuracy."""
    data.set_seed()
    train_loader, val_loader, test_loader = data.get_dataloaders()

    model = MLPClassifier().to(configs.DEVICE)
    optimizer = Adam(model.parameters(), lr=configs.MLP_LR)
    history = utils.TrainingHistory(MODEL_NAME)

    # The epoch is chosen on the validation set so the test set stays untouched.
    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(1, configs.MLP_EPOCHS + 1):
        with utils.Timer() as timer:
            train_loss = train_one_epoch(model, train_loader, optimizer)
        val_loss, val_acc = evaluate(model, val_loader)
        history.record(
            {"train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc},
            timer.elapsed,
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
        print(
            f"epoch {epoch:02d}/{configs.MLP_EPOCHS} | train loss {train_loss:.4f} | "
            f"val loss {val_loss:.4f} | val acc {val_acc:.4f} | {timer.elapsed:.1f}s"
        )

    model.load_state_dict(best_state)
    utils.save_checkpoint(model, MODEL_NAME)
    history.save()

    test_loss, test_acc = evaluate(model, test_loader)
    target_met = test_acc >= configs.MLP_TARGET_ACC
    utils.save_json(
        {
            "model": MODEL_NAME,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "target_acc": configs.MLP_TARGET_ACC,
            "target_met": target_met,
        },
        configs.RESULT_DIR / f"{MODEL_NAME}.json",
    )
    print(f"test acc {test_acc:.4f} (target {configs.MLP_TARGET_ACC:.2f}, met: {target_met})")


if __name__ == "__main__":
    main()
