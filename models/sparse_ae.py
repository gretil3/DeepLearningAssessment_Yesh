"""Sparse autoencoder: a 256-unit ReLU code that is pushed towards sparsity by an L1 penalty."""

import torch
from torch import nn

import configs


class SparseAutoencoder(nn.Module):
    """Single-hidden-layer autoencoder: 784 -> 256 (sparse code h) -> 784.

    The model itself is a plain autoencoder; sparsity comes from the L1 penalty on the
    code ``h`` that is added to the loss in ``train/train_sparse_ae.py``.
    """

    def __init__(self, hidden_dim: int = configs.SPARSE_HIDDEN_DIM) -> None:
        """Build the encoder and decoder.

        Args:
            hidden_dim: Size of the sparse code h.
        """
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(configs.INPUT_DIM, hidden_dim),
            nn.ReLU(),  # h >= 0, so ||h||_1 is simply the sum of the activations
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, configs.INPUT_DIM),
            nn.Sigmoid(),  # pixels live in [0, 1]
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Map images to the sparse code h.

        Args:
            x: Image batch of shape (B, 1, 28, 28).

        Returns:
            Non-negative code of shape (B, 256).
        """
        return self.encoder(x)

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        """Map codes back to images.

        Args:
            h: Code batch of shape (B, 256).

        Returns:
            Images of shape (B, 1, 28, 28) with values in [0, 1].
        """
        return self.decoder(h).reshape(-1, 1, configs.IMAGE_SIZE, configs.IMAGE_SIZE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct images.

        Args:
            x: Image batch of shape (B, 1, 28, 28).

        Returns:
            Reconstructions of shape (B, 1, 28, 28).
        """
        return self.decode(self.encode(x))
