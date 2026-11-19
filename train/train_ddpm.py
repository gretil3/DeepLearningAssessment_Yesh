"""Train the DDPM to predict the added noise, keeping an EMA copy of the UNet for sampling.

Run from the repository root: ``python -m train.train_ddpm``. On a CPU one epoch takes
several minutes, so for the full ``DDPM_EPOCHS`` a GPU (for example Colab) is advisable.
"""

import copy

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

import configs
import data
import utils
from models.ddpm import GaussianDiffusion

MODEL_NAME = "ddpm"


@torch.no_grad()
def update_ema(ema: nn.Module, model: nn.Module, decay: float) -> None:
    """Move the EMA weights a small step towards the current weights.

    Args:
        ema: Module holding the exponential moving average.
        model: Module being trained.
        decay: EMA decay; closer to 1 averages over more steps.
    """
    # ema <- decay * ema + (1 - decay) * weights
    for p_ema, p in zip(ema.parameters(), model.parameters()):
        p_ema.mul_(decay).add_(p, alpha=1.0 - decay)


@torch.no_grad()
def validation_loss(diffusion: GaussianDiffusion, loader: DataLoader) -> float:
    """Mean noise-prediction loss on a loader, with fixed random steps and noise.

    Each call draws the same random numbers, so values from different epochs compare
    fairly. The training random stream is restored afterwards.

    Args:
        diffusion: Diffusion model to score.
        loader: Validation dataloader.

    Returns:
        Mean loss over the loader.
    """
    diffusion.eval()
    cuda_devices = list(range(torch.cuda.device_count())) if configs.DEVICE.type == "cuda" else []
    total_loss, total_items = 0.0, 0
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(configs.SEED)
        for images, _ in loader:
            images = images.to(configs.DEVICE)
            total_loss += diffusion.loss(images).item() * images.size(0)
            total_items += images.size(0)
    return total_loss / total_items


def main() -> None:
    """Train the diffusion model and save the EMA weights as the checkpoint."""
    data.set_seed()
    train_loader, val_loader, _ = data.get_dataloaders()

    diffusion = GaussianDiffusion().to(configs.DEVICE)
    ema_unet = copy.deepcopy(diffusion.unet).requires_grad_(False)
    optimizer = Adam(diffusion.parameters(), lr=configs.DDPM_LR)
    history = utils.TrainingHistory(MODEL_NAME)

    for epoch in range(1, configs.DDPM_EPOCHS + 1):
        with utils.Timer() as timer:
            diffusion.train()
            total_loss, total_items = 0.0, 0
            for images, _ in train_loader:
                images = images.to(configs.DEVICE)
                loss = diffusion.loss(images)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(diffusion.parameters(), configs.DDPM_GRAD_CLIP)
                optimizer.step()
                update_ema(ema_unet, diffusion.unet, configs.DDPM_EMA_DECAY)
                # Weight by batch size because the last batch is smaller.
                total_loss += loss.item() * images.size(0)
                total_items += images.size(0)
        train_loss = total_loss / total_items
        val_loss = validation_loss(diffusion, val_loader)
        history.record({"train_loss": train_loss, "val_loss": val_loss}, timer.elapsed)
        print(
            f"epoch {epoch:02d}/{configs.DDPM_EPOCHS} | train loss {train_loss:.5f} | "
            f"val loss {val_loss:.5f} | {timer.elapsed:.1f}s"
        )

    # Final weights are used as they are: the noisy diffusion loss is a poor selection signal.
    diffusion.unet.load_state_dict(ema_unet.state_dict())
    utils.save_checkpoint(diffusion, MODEL_NAME)
    history.save()


if __name__ == "__main__":
    main()
