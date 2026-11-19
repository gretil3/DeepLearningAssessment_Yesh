"""Small time-conditioned UNet that predicts the noise added to an image."""

import math

import torch
from torch import nn

import configs


class SinusoidalEmbedding(nn.Module):
    """Sinusoidal embedding of the integer diffusion step, as in the Transformer."""

    def __init__(self, dim: int) -> None:
        """Store the output size.

        Args:
            dim: Embedding size (must be even).
        """
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Embed a batch of time steps.

        Args:
            t: Integer time steps of shape (B,).

        Returns:
            Embeddings of shape (B, dim).
        """
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
        angles = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([angles.sin(), angles.cos()], dim=1)


class ResBlock(nn.Module):
    """Residual block with GroupNorm, SiLU, and an additive time-step embedding."""

    def __init__(self, in_channels: int, out_channels: int, emb_dim: int, groups: int) -> None:
        """Build the block.

        Args:
            in_channels: Input channel count.
            out_channels: Output channel count.
            emb_dim: Size of the time embedding.
            groups: GroupNorm groups.
        """
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.emb_proj = nn.Linear(emb_dim, out_channels)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """Apply the block.

        Args:
            x: Feature map of shape (B, in_channels, H, W).
            emb: Time embedding of shape (B, emb_dim).

        Returns:
            Feature map of shape (B, out_channels, H, W).
        """
        h = self.conv1(torch.nn.functional.silu(self.norm1(x)))
        h = h + self.emb_proj(torch.nn.functional.silu(emb))[:, :, None, None]
        h = self.conv2(torch.nn.functional.silu(self.norm2(h)))
        return h + self.skip(x)


class UNet(nn.Module):
    """UNet with three resolutions (28, 14, 7) and skip connections; predicts epsilon."""

    def __init__(
        self,
        base_channels: int = configs.DDPM_BASE_CHANNELS,
        emb_dim: int = configs.DDPM_TIME_EMBED_DIM,
        groups: int = configs.DDPM_GROUPS,
    ) -> None:
        """Build the encoder, bottleneck, and decoder.

        Args:
            base_channels: Channels at 28x28; the 14x14 and 7x7 maps use twice as many.
            emb_dim: Size of the time embedding.
            groups: GroupNorm groups.
        """
        super().__init__()
        c = base_channels
        self.time_embed = nn.Sequential(
            SinusoidalEmbedding(c), nn.Linear(c, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim)
        )
        self.conv_in = nn.Conv2d(1, c, 3, padding=1)
        # Encoder: 28x28 -> 14x14 -> 7x7
        self.down1 = ResBlock(c, c, emb_dim, groups)
        self.pool1 = nn.Conv2d(c, c, 3, stride=2, padding=1)
        self.down2 = ResBlock(c, 2 * c, emb_dim, groups)
        self.pool2 = nn.Conv2d(2 * c, 2 * c, 3, stride=2, padding=1)
        self.down3 = ResBlock(2 * c, 2 * c, emb_dim, groups)
        self.mid = ResBlock(2 * c, 2 * c, emb_dim, groups)
        # Decoder: 7x7 -> 14x14 -> 28x28; inputs are concatenated with the encoder skips
        self.up3 = ResBlock(4 * c, 2 * c, emb_dim, groups)
        self.upsample2 = nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(2 * c, 2 * c, 3, padding=1))
        self.up2 = ResBlock(4 * c, 2 * c, emb_dim, groups)
        self.upsample1 = nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(2 * c, 2 * c, 3, padding=1))
        self.up1 = ResBlock(3 * c, c, emb_dim, groups)
        self.out = nn.Sequential(nn.GroupNorm(groups, c), nn.SiLU(), nn.Conv2d(c, 1, 3, padding=1))

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict the noise contained in ``x`` at diffusion step ``t``.

        Args:
            x: Noisy images of shape (B, 1, 28, 28).
            t: Integer diffusion steps of shape (B,).

        Returns:
            Predicted noise of shape (B, 1, 28, 28).
        """
        emb = self.time_embed(t)
        s1 = self.down1(self.conv_in(x), emb)
        s2 = self.down2(self.pool1(s1), emb)
        s3 = self.down3(self.pool2(s2), emb)
        h = self.up3(torch.cat([self.mid(s3, emb), s3], dim=1), emb)
        h = self.up2(torch.cat([self.upsample2(h), s2], dim=1), emb)
        h = self.up1(torch.cat([self.upsample1(h), s1], dim=1), emb)
        return self.out(h)
