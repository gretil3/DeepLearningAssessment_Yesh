"""Variational autoencoder with the same decoder as the Vanilla Autoencoder."""

import torch
from torch import nn

import configs
from models.vanilla_ae import MLPDecoder


class VAE(nn.Module):
    """VAE: 784 -> 256 -> (mu, logvar) of size 64 -> 256 -> 784.

    The decoder is the exact :class:`MLPDecoder` used by the Vanilla Autoencoder, so any
    difference between the two models comes from the probabilistic encoder and the KL term.
    """

    def __init__(
        self,
        latent_dim: int = configs.VAE_LATENT_DIM,
        hidden_dim: int = configs.VAE_HIDDEN_DIM,
    ) -> None:
        """Build the encoder heads and the decoder.

        Args:
            latent_dim: Size of the latent variable z.
            hidden_dim: Size of the hidden layer on each side.
        """
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Flatten(),
            nn.Linear(configs.INPUT_DIM, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        # Predicting log(sigma^2) keeps the variance positive without a constraint.
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = MLPDecoder(latent_dim, hidden_dim)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the parameters of the approximate posterior q(z | x).

        Args:
            x: Image batch of shape (B, 1, 28, 28).

        Returns:
            A tuple (mu, logvar), each of shape (B, 64).
        """
        hidden = self.trunk(x)
        return self.fc_mu(hidden), self.fc_logvar(hidden)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Draw z = mu + sigma * eps with eps ~ N(0, I), keeping gradients to mu and sigma.

        Args:
            mu: Posterior means, shape (B, 64).
            logvar: Posterior log-variances, shape (B, 64).

        Returns:
            Latent samples of shape (B, 64).
        """
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Map latent vectors to images.

        Args:
            z: Latent batch of shape (B, 64).

        Returns:
            Images of shape (B, 1, 28, 28) with values in [0, 1].
        """
        return self.decoder(z)

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct images deterministically from the posterior mean.

        Args:
            x: Image batch of shape (B, 1, 28, 28).

        Returns:
            Reconstructions of shape (B, 1, 28, 28).
        """
        mu, _ = self.encode(x)
        return self.decode(mu)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode, sample, and decode.

        Args:
            x: Image batch of shape (B, 1, 28, 28).

        Returns:
            A tuple (reconstruction, mu, logvar).
        """
        mu, logvar = self.encode(x)
        return self.decode(self.reparameterize(mu, logvar)), mu, logvar
