"""Vanilla GAN: fully connected generator and discriminator."""

import torch
from torch import nn

import configs


class Generator(nn.Module):
    """MLP generator mapping noise z (100) to a 28x28 image: 100 -> 256 -> 512 -> 1024 -> 784."""

    def __init__(
        self,
        latent_dim: int = configs.GAN_LATENT_DIM,
        hidden_dims: tuple[int, int, int] = configs.GAN_HIDDEN_DIMS,
    ) -> None:
        """Build the layers.

        Args:
            latent_dim: Size of the input noise vector.
            hidden_dims: Widths of the three hidden layers.
        """
        super().__init__()
        h1, h2, h3 = hidden_dims
        self.net = nn.Sequential(
            nn.Linear(latent_dim, h1), nn.LeakyReLU(configs.GAN_LEAKY_SLOPE),
            nn.Linear(h1, h2), nn.LeakyReLU(configs.GAN_LEAKY_SLOPE),
            nn.Linear(h2, h3), nn.LeakyReLU(configs.GAN_LEAKY_SLOPE),
            nn.Linear(h3, configs.INPUT_DIM),
            nn.Sigmoid(),  # pixels live in [0, 1], like every other model in this project
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Generate images from noise.

        Args:
            z: Noise batch of shape (B, 100).

        Returns:
            Images of shape (B, 1, 28, 28) with values in [0, 1].
        """
        return self.net(z).reshape(-1, 1, configs.IMAGE_SIZE, configs.IMAGE_SIZE)


class Discriminator(nn.Module):
    """MLP discriminator: 784 -> 1024 -> 512 -> 256 -> 1 (a logit, not a probability)."""

    def __init__(self, hidden_dims: tuple[int, int, int] = configs.GAN_HIDDEN_DIMS) -> None:
        """Build the layers, mirroring the generator widths in reverse.

        Args:
            hidden_dims: Generator widths; the discriminator uses them in reverse order.
        """
        super().__init__()
        h1, h2, h3 = hidden_dims
        slope, drop = configs.GAN_LEAKY_SLOPE, configs.GAN_DROPOUT
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(configs.INPUT_DIM, h3), nn.LeakyReLU(slope), nn.Dropout(drop),
            nn.Linear(h3, h2), nn.LeakyReLU(slope), nn.Dropout(drop),
            nn.Linear(h2, h1), nn.LeakyReLU(slope), nn.Dropout(drop),
            nn.Linear(h1, 1),  # logit; the sigmoid is folded into the BCE-with-logits loss
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Score images as real or fake.

        Args:
            x: Image batch of shape (B, 1, 28, 28).

        Returns:
            Logits of shape (B,); a higher value means "more real".
        """
        return self.net(x).squeeze(1)
