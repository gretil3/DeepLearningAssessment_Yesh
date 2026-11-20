"""Image-quality metrics: MSE, PSNR, SSIM over a dataloader, and the FID of generated images."""

from collections.abc import Callable

import torch
from torch.utils.data import DataLoader
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.fid import FrechetInceptionDistance

import configs


@torch.no_grad()
def reconstruction_metrics(
    reconstruct: Callable[[torch.Tensor], torch.Tensor], loader: DataLoader
) -> dict[str, float]:
    """Score a reconstruction function against the original images.

    Images are in [0, 1], N is the number of images, and D = 784 pixels per image:
        MSE  = (1 / (N * D)) * sum_i ||x_i - x_hat_i||^2
        PSNR = mean_i [ 10 * log10(1 / MSE_i) ]   (per image, peak value 1)
        SSIM = mean_i SSIM(x_i, x_hat_i)          (11x11 Gaussian window, sigma 1.5)

    Args:
        reconstruct: Maps an image batch (B, 1, 28, 28) to reconstructions with the
            same number of elements, in [0, 1].
        loader: Dataloader holding the images to score (the test loader).

    Returns:
        Dict with the keys ``mse``, ``psnr`` (dB), and ``ssim``.
    """
    # PSNR is averaged per image (dim=(1, 2, 3)), not computed from the pooled MSE.
    psnr = PeakSignalNoiseRatio(data_range=1.0, dim=(1, 2, 3)).to(configs.DEVICE)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(configs.DEVICE)
    squared_error, num_values = 0.0, 0
    for images, _ in loader:
        images = images.to(configs.DEVICE)
        recon = reconstruct(images).reshape_as(images)
        squared_error += (images - recon).pow(2).sum().item()
        num_values += images.numel()
        psnr.update(recon, images)
        ssim.update(recon, images)
    return {
        "mse": squared_error / num_values,
        "psnr": psnr.compute().item(),
        "ssim": ssim.compute().item(),
    }


class FidScorer:
    """Frechet Inception Distance of generated images against a fixed set of real images.

    The Inception features of the real images are computed once and reused for every
    model that is scored. Images are in [0, 1]; the single grey channel is repeated to
    three channels and resized to 299x299 inside the Inception network.

    Note:
        The Inception weights (about 90 MB) are downloaded by ``torch-fidelity`` on the
        first use.
    """

    def __init__(self, real_images: torch.Tensor) -> None:
        """Compute the real-image feature statistics.

        Args:
            real_images: Real images of shape (N, 1, 28, 28) in [0, 1] (the test set).
        """
        # reset_real_features=False keeps the real statistics when the metric is reset.
        self._fid = FrechetInceptionDistance(
            feature=2048, normalize=True, reset_real_features=False
        ).to(configs.DEVICE)
        self._update(real_images, real=True)

    @torch.no_grad()
    def _update(self, images: torch.Tensor, real: bool) -> None:
        """Feed images through Inception in batches.

        Args:
            images: Image tensor of shape (N, 1, 28, 28) in [0, 1].
            real: True for real images, False for generated ones.
        """
        for batch in images.split(configs.BATCH_SIZE):
            rgb = batch.to(configs.DEVICE).clamp(0.0, 1.0).repeat(1, 3, 1, 1)
            self._fid.update(rgb, real=real)

    def score(self, fake_images: torch.Tensor) -> float:
        """Compute the FID of generated images (lower is better).

        FID = ||mu_r - mu_g||^2 + Tr(S_r + S_g - 2 (S_r S_g)^(1/2)), where (mu, S) are the
        mean and covariance of the 2048-d Inception features of the real (r) and
        generated (g) images.

        Args:
            fake_images: Generated images of shape (n, 1, 28, 28) in [0, 1].

        Returns:
            The FID value.
        """
        self._update(fake_images, real=False)
        value = self._fid.compute().item()
        self._fid.reset()  # drops the generated statistics only
        return value
