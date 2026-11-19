"""DCGAN: fully convolutional generator and discriminator (strided convs, BatchNorm, no pooling)."""

import torch
from torch import nn

import configs


def _init_weights(module: nn.Module) -> None:
    """Initialise conv and batch-norm weights as recommended for DCGAN.

    Args:
        module: Layer visited by ``nn.Module.apply``.
    """
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, 0.0, configs.DCGAN_INIT_STD)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight, 1.0, configs.DCGAN_INIT_STD)
        nn.init.zeros_(module.bias)


class DCGANGenerator(nn.Module):
    """Transposed-conv generator: z (100) -> 128x7x7 -> 64x14x14 -> 1x28x28."""

    def __init__(
        self,
        latent_dim: int = configs.DCGAN_LATENT_DIM,
        base_channels: int = configs.DCGAN_BASE_CHANNELS,
    ) -> None:
        """Build the layers.

        Args:
            latent_dim: Size of the input noise vector.
            base_channels: Channel count of the 14x14 feature map; the 7x7 map has twice as many.
        """
        super().__init__()
        c = base_channels
        self.net = nn.Sequential(
            # 1x1 -> 7x7: the noise vector is treated as a 1x1 feature map
            nn.ConvTranspose2d(latent_dim, 2 * c, kernel_size=7, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(2 * c), nn.ReLU(),
            # 7x7 -> 14x14
            nn.ConvTranspose2d(2 * c, c, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c), nn.ReLU(),
            # 14x14 -> 28x28
            nn.ConvTranspose2d(c, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),  # pixels live in [0, 1], like every other model in this project
        )
        self.apply(_init_weights)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Generate images from noise.

        Args:
            z: Noise batch of shape (B, 100).

        Returns:
            Images of shape (B, 1, 28, 28) with values in [0, 1].
        """
        return self.net(z.reshape(z.size(0), -1, 1, 1))


class DCGANDiscriminator(nn.Module):
    """Strided-conv discriminator: 1x28x28 -> 64x14x14 -> 128x7x7 -> 1 logit."""

    def __init__(self, base_channels: int = configs.DCGAN_BASE_CHANNELS) -> None:
        """Build the layers.

        Args:
            base_channels: Channel count of the first feature map; the second has twice as many.
        """
        super().__init__()
        c = base_channels
        slope = configs.DCGAN_LEAKY_SLOPE
        self.net = nn.Sequential(
            # No BatchNorm on the input layer, as in the DCGAN guidelines
            nn.Conv2d(1, c, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(slope),
            nn.Conv2d(c, 2 * c, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(2 * c), nn.LeakyReLU(slope),
            # 7x7 -> 1x1 logit; a conv with a 7x7 kernel replaces a fully connected layer
            nn.Conv2d(2 * c, 1, kernel_size=7, stride=1, padding=0),
        )
        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Score images as real or fake.

        Args:
            x: Image batch of shape (B, 1, 28, 28).

        Returns:
            Logits of shape (B,); a higher value means "more real".
        """
        return self.net(x).reshape(-1)
