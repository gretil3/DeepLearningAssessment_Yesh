"""MLP classifier whose weights and latent features are probed in the transposition experiments."""

import torch
from torch import nn

import configs


class MLPClassifier(nn.Module):
    """Bias-free MLP classifier: 784 -> 256 -> 64 -> 10.

    Biases are disabled so every layer is a pure matrix product. This keeps the
    transpose projection W^T used in the experiments exact (there is no offset to undo).

    Note:
        ``nn.Linear`` stores its weight as (out_features, in_features), so
        ``fc1.weight`` is W1 with shape (256, 784) in the assignment's notation
        and ``fc1.weight.T`` is its transpose.
    """

    def __init__(
        self,
        input_dim: int = configs.INPUT_DIM,
        hidden_dims: tuple[int, int] = configs.MLP_HIDDEN_DIMS,
        num_classes: int = configs.NUM_CLASSES,
    ) -> None:
        """Build the three linear layers.

        Args:
            input_dim: Number of flattened input pixels.
            hidden_dims: Sizes of the two hidden layers; the second one is the latent size.
            num_classes: Number of output classes.
        """
        super().__init__()
        h1, h2 = hidden_dims
        self.fc1 = nn.Linear(input_dim, h1, bias=False)
        self.fc2 = nn.Linear(h1, h2, bias=False)
        self.fc3 = nn.Linear(h2, num_classes, bias=False)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the latent feature z_MLP = ReLU(W2 ReLU(W1 x)).

        Args:
            x: Image batch of shape (B, 1, 28, 28) or (B, 784).

        Returns:
            Latent features of shape (B, 64), taken before the final classification layer.
        """
        x = x.flatten(start_dim=1)
        return torch.relu(self.fc2(torch.relu(self.fc1(x))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute class logits.

        Args:
            x: Image batch of shape (B, 1, 28, 28) or (B, 784).

        Returns:
            Unnormalised class scores of shape (B, 10).
        """
        return self.fc3(self.encode(x))
