"""Vanilla autoencoder and the MLP decoder it shares with the frozen-latent experiment."""

import torch
from torch import nn

import configs


class MLPDecoder(nn.Module):
    """Decoder 64 -> 256 -> 784 with a sigmoid output.

    Used both by :class:`VanillaAutoencoder` and by the frozen-latent decoding experiment,
    so the two are compared with the exact same decoder architecture.
    """

    def __init__(
        self,
        latent_dim: int = configs.AE_LATENT_DIM,
        hidden_dim: int = configs.AE_HIDDEN_DIM,
        output_dim: int = configs.INPUT_DIM,
    ) -> None:
        """Build the two linear layers.

        Args:
            latent_dim: Size of the latent input vector.
            hidden_dim: Size of the hidden layer.
            output_dim: Number of output pixels (flattened image).
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),  # pixels live in [0, 1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vectors to images.

        Args:
            z: Latent batch of shape (B, 64).

        Returns:
            Images of shape (B, 1, 28, 28) with values in [0, 1].
        """
        return self.net(z).reshape(-1, 1, configs.IMAGE_SIZE, configs.IMAGE_SIZE)


class VanillaAutoencoder(nn.Module):
    """Deterministic autoencoder: 784 -> 256 -> 64 (latent) -> 256 -> 784."""

    def __init__(
        self,
        latent_dim: int = configs.AE_LATENT_DIM,
        hidden_dim: int = configs.AE_HIDDEN_DIM,
    ) -> None:
        """Build the encoder and decoder.

        Args:
            latent_dim: Size of the bottleneck z_AE.
            hidden_dim: Size of the hidden layer on each side.
        """
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(configs.INPUT_DIM, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),  # linear latent: no activation on z_AE
        )
        self.decoder = MLPDecoder(latent_dim, hidden_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Map images to latent vectors z_AE.

        Args:
            x: Image batch of shape (B, 1, 28, 28).

        Returns:
            Latent batch of shape (B, 64).
        """
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Map latent vectors back to images.

        Args:
            z: Latent batch of shape (B, 64).

        Returns:
            Images of shape (B, 1, 28, 28).
        """
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct images.

        Args:
            x: Image batch of shape (B, 1, 28, 28).

        Returns:
            Reconstructions of shape (B, 1, 28, 28).
        """
        return self.decode(self.encode(x))
