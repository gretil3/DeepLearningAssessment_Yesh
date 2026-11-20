"""Load trained checkpoints into ready-to-use autoencoders and prior samplers."""

from collections.abc import Callable

import torch
from torch import nn

import configs
import utils
from models.dcgan import DCGANGenerator
from models.ddpm import GaussianDiffusion
from models.gan import Generator
from models.sparse_ae import SparseAutoencoder
from models.vae import VAE
from models.vanilla_ae import VanillaAutoencoder

AUTOENCODER_NAMES = ("vanilla_ae", "sparse_ae", "vae")
GENERATOR_NAMES = ("vae", "vanilla_gan", "dcgan", "ddpm")
# Checkpoint that holds the weights used for sampling (a GAN's generator is saved separately).
GENERATOR_CHECKPOINTS = {
    "vae": "vae",
    "vanilla_gan": "vanilla_gan_generator",
    "dcgan": "dcgan_generator",
    "ddpm": "ddpm",
}


def load_autoencoder(name: str) -> nn.Module:
    """Load a trained autoencoder in eval mode.

    Args:
        name: One of ``AUTOENCODER_NAMES``.

    Returns:
        The model on ``configs.DEVICE``.
    """
    builders = {"vanilla_ae": VanillaAutoencoder, "sparse_ae": SparseAutoencoder, "vae": VAE}
    return utils.load_checkpoint(builders[name](), name)


def load_decoder(name: str) -> tuple[Callable[[torch.Tensor], torch.Tensor], int]:
    """Return the latent-to-image network of a single-pass generator.

    Args:
        name: ``"vae"``, ``"vanilla_gan"``, or ``"dcgan"``.

    Returns:
        A tuple (decode, latent_dim): ``decode`` maps latent vectors (B, latent_dim) to
        images (B, 1, 28, 28) in [0, 1], on ``configs.DEVICE``.
    """
    if name == "vae":
        return load_autoencoder("vae").decode, configs.VAE_LATENT_DIM
    if name == "vanilla_gan":
        return utils.load_checkpoint(Generator(), GENERATOR_CHECKPOINTS[name]), configs.GAN_LATENT_DIM
    if name == "dcgan":
        return utils.load_checkpoint(DCGANGenerator(), GENERATOR_CHECKPOINTS[name]), configs.DCGAN_LATENT_DIM
    raise ValueError(f"unknown single-pass generator: {name}")


def load_prior_sampler(name: str) -> Callable[[int], torch.Tensor]:
    """Return a function that draws new images from a model's prior.

    Args:
        name: One of ``GENERATOR_NAMES``.

    Returns:
        ``sample(n)`` giving ``n`` images of shape (n, 1, 28, 28) in [0, 1] on
        ``configs.DEVICE``. VAE, Vanilla GAN, and DCGAN decode z ~ N(0, I); the DDPM runs
        the full reverse process from pure noise.
    """
    if name == "ddpm":
        return load_diffusion().sample

    decode, latent_dim = load_decoder(name)

    @torch.no_grad()
    def sample(num_samples: int) -> torch.Tensor:
        z = torch.randn(num_samples, latent_dim, device=configs.DEVICE)
        return decode(z)

    return sample


def load_diffusion() -> GaussianDiffusion:
    """Load the trained DDPM (EMA weights) in eval mode."""
    return utils.load_checkpoint(GaussianDiffusion(), GENERATOR_CHECKPOINTS["ddpm"])
