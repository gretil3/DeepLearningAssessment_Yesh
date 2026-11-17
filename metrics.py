"""Image-quality metrics (MSE, PSNR, SSIM) computed over a dataloader."""

from collections.abc import Callable

import torch
from torch.utils.data import DataLoader
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

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
