"""Shared adversarial training loop for the Vanilla GAN and the DCGAN."""

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

import configs
import utils


def train_one_epoch(
    generator: nn.Module,
    discriminator: nn.Module,
    loader: DataLoader,
    opt_g: Adam,
    opt_d: Adam,
    latent_dim: int,
) -> dict[str, float]:
    """Run one epoch of alternating discriminator and generator updates.

    Args:
        generator: Maps noise (B, latent_dim) to images.
        discriminator: Maps images to logits (B,).
        loader: Training dataloader (labels are ignored).
        opt_g: Optimizer of the generator.
        opt_d: Optimizer of the discriminator.
        latent_dim: Size of the noise vector.

    Returns:
        Epoch means of ``d_loss``, ``g_loss``, ``d_real`` (D's mean probability on real
        images) and ``d_fake`` (D's mean probability on generated images).
    """
    generator.train()
    discriminator.train()
    sums = {"d_loss": 0.0, "g_loss": 0.0, "d_real": 0.0, "d_fake": 0.0}
    total_items = 0
    for real, _ in loader:
        real = real.to(configs.DEVICE)
        batch_size = real.size(0)
        ones = torch.ones(batch_size, device=configs.DEVICE)
        zeros = torch.zeros(batch_size, device=configs.DEVICE)
        fake = generator(torch.randn(batch_size, latent_dim, device=configs.DEVICE))

        # Discriminator step. With D(x) = sigmoid(logit):
        #   L_D = -E_x[log D(x)] - E_z[log(1 - D(G(z)))]
        # G is detached so this step only updates D.
        real_logits = discriminator(real)
        fake_logits = discriminator(fake.detach())
        d_loss = (
            F.binary_cross_entropy_with_logits(real_logits, ones)
            + F.binary_cross_entropy_with_logits(fake_logits, zeros)
        )
        opt_d.zero_grad()
        d_loss.backward()
        opt_d.step()

        # Generator step, non-saturating form: L_G = -E_z[log D(G(z))].
        # The original E_z[log(1 - D(G(z)))] has vanishing gradients while D is winning.
        g_loss = F.binary_cross_entropy_with_logits(discriminator(fake), ones)
        opt_g.zero_grad()
        g_loss.backward()
        opt_g.step()

        # Weight by batch size because the last batch is smaller.
        sums["d_loss"] += d_loss.item() * batch_size
        sums["g_loss"] += g_loss.item() * batch_size
        sums["d_real"] += torch.sigmoid(real_logits).sum().item()
        sums["d_fake"] += torch.sigmoid(fake_logits).sum().item()
        total_items += batch_size
    return {key: value / total_items for key, value in sums.items()}


def fit_gan(
    generator: nn.Module,
    discriminator: nn.Module,
    name: str,
    train_loader: DataLoader,
    latent_dim: int,
    lr: float,
    beta1: float,
    epochs: int,
) -> utils.TrainingHistory:
    """Train a GAN and save both networks and the per-epoch log.

    There is no validation-based model selection: GAN losses do not measure sample
    quality, so the weights of the final epoch are kept. Writes
    ``outputs/checkpoints/<name>_generator.pt``, ``<name>_discriminator.pt`` and the
    log ``outputs/logs/<name>.json``.

    Args:
        generator: Generator already moved to ``configs.DEVICE``.
        discriminator: Discriminator already moved to ``configs.DEVICE``.
        name: Run identifier used for the output file names.
        train_loader: Training dataloader.
        latent_dim: Size of the noise vector.
        lr: Adam learning rate (shared by both networks).
        beta1: Adam beta1 (shared by both networks).
        epochs: Number of epochs.

    Returns:
        The recorded training history.
    """
    opt_g = Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
    opt_d = Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
    history = utils.TrainingHistory(name)

    for epoch in range(1, epochs + 1):
        with utils.Timer() as timer:
            stats = train_one_epoch(generator, discriminator, train_loader, opt_g, opt_d, latent_dim)
        history.record(stats, timer.elapsed)
        print(
            f"epoch {epoch:02d}/{epochs} | D loss {stats['d_loss']:.3f} | G loss {stats['g_loss']:.3f} | "
            f"D(x) {stats['d_real']:.2f} | D(G(z)) {stats['d_fake']:.2f} | {timer.elapsed:.1f}s"
        )

    utils.save_checkpoint(generator, f"{name}_generator")
    utils.save_checkpoint(discriminator, f"{name}_discriminator")
    history.save()
    return history
