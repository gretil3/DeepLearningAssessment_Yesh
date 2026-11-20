"""Denoising diffusion probabilistic model (DDPM) with a linear noise schedule."""

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

import configs
from models.unet import UNet


class GaussianDiffusion(nn.Module):
    """DDPM: forward noising, noise-prediction loss, and ancestral sampling.

    Public methods take and return images in [0, 1] like every other model in this
    project; the rescaling to [-1, 1] that diffusion needs happens inside.
    """

    def __init__(self) -> None:
        """Build the UNet and the noise-schedule buffers (saved with the checkpoint)."""
        super().__init__()
        self.unet = UNet()
        self.timesteps = configs.DDPM_TIMESTEPS
        betas = torch.linspace(configs.DDPM_BETA_START, configs.DDPM_BETA_END, self.timesteps)
        alphas = 1.0 - betas
        # float64 cumprod avoids rounding drift over 1000 steps
        alpha_bars = torch.cumprod(alphas.double(), dim=0).float()
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)

    def loss(self, images: torch.Tensor) -> torch.Tensor:
        """Noise-prediction loss of a batch.

        Args:
            images: Image batch of shape (B, 1, 28, 28) in [0, 1].

        Returns:
            Scalar loss.
        """
        # Forward process: q(x_t | x_0) = N(sqrt(abar_t) x_0, (1 - abar_t) I), i.e.
        #   x_t = sqrt(abar_t) x_0 + sqrt(1 - abar_t) eps,  eps ~ N(0, I)
        # Loss (Ho et al., "simple" form): L = E_{t, x_0, eps} || eps - eps_theta(x_t, t) ||^2
        x0 = images * 2.0 - 1.0
        t = torch.randint(0, self.timesteps, (x0.size(0),), device=x0.device)
        eps = torch.randn_like(x0)
        abar = self.alpha_bars[t].view(-1, 1, 1, 1)
        x_t = abar.sqrt() * x0 + (1.0 - abar).sqrt() * eps
        return F.mse_loss(self.unet(x_t, t), eps)

    @torch.no_grad()
    def sample(self, num_samples: int) -> torch.Tensor:
        """Generate images by running the reverse process from pure noise.

        Args:
            num_samples: Number of images to generate.

        Returns:
            Images of shape (num_samples, 1, 28, 28) in [0, 1].
        """
        return self.sample_trajectory(num_samples, ())[0]

    @torch.no_grad()
    def sample_trajectory(
        self, num_samples: int, snapshot_steps: Sequence[int]
    ) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
        """Run the reverse process and keep intermediate states for visualisation.

        The state x_t is indexed so that t = ``timesteps`` is the initial pure noise and
        t = 0 is the final image.

        Args:
            num_samples: Number of images to generate.
            snapshot_steps: Values of t (between 0 and ``timesteps``) whose state is kept.

        Returns:
            A tuple (images, snapshots): the final images of shape (num_samples, 1, 28, 28)
            in [0, 1], and a dict mapping each requested t to its state, clamped to the
            same range for display.
        """
        device = self.betas.device
        wanted = set(snapshot_steps)
        snapshots: dict[int, torch.Tensor] = {}

        def keep(t: int, state: torch.Tensor) -> None:
            if t in wanted:
                snapshots[t] = (state.clamp(-1.0, 1.0) + 1.0) / 2.0

        x = torch.randn(num_samples, 1, configs.IMAGE_SIZE, configs.IMAGE_SIZE, device=device)
        keep(self.timesteps, x)
        for step in reversed(range(self.timesteps)):
            t = torch.full((num_samples,), step, device=device, dtype=torch.long)
            eps = self.unet(x, t)
            # Ancestral step, with sigma_t^2 = beta_t:
            #   x_{t-1} = (x_t - beta_t / sqrt(1 - abar_t) * eps_theta) / sqrt(alpha_t) + sigma_t * z
            mean = (x - self.betas[step] / (1.0 - self.alpha_bars[step]).sqrt() * eps) / self.alphas[step].sqrt()
            noise = torch.randn_like(x) if step > 0 else torch.zeros_like(x)  # no noise at the last step
            x = mean + self.betas[step].sqrt() * noise
            keep(step, x)
        return (x.clamp(-1.0, 1.0) + 1.0) / 2.0, snapshots
