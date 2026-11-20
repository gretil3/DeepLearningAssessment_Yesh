# Comparative Study of Generative Models (Deep Learning D6421, Session 07)

A PyTorch study that compares six generative models on Fashion-MNIST: Vanilla
Autoencoder, Sparse Autoencoder, VAE, Vanilla GAN, DCGAN, and DDPM. An MLP
classifier is also trained to show why discriminative features cannot be used
directly for image generation.

All models share the same data split, batch size, and random seed (42), and every
metric is computed on the held-out test set.

> Work in progress: modules are added one at a time. This README is updated with each one.

## Installation

Requires Python 3.10+.

```bash
python -m venv .venv
.venv\Scripts\activate        # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Fashion-MNIST is downloaded automatically by `torchvision` into `./data` on first use.
The code picks CUDA when available and falls back to CPU, so it runs locally and on Colab.

## Usage

Quick check of the data pipeline (prints the size of each split):

```bash
python -c "import data; tr, va, te = data.get_dataloaders(); print(len(tr.dataset), len(va.dataset), len(te.dataset))"
```

Expected output: `56000 7000 7000`.

Training scripts are run as modules from the repository root. Each script saves its
checkpoint, per-epoch loss, and per-epoch training time.

| Command | What it does | Outputs |
|---|---|---|
| `python -m train.train_mlp` | Trains the 784→256→64→10 MLP classifier (target test accuracy ≥ 88%) | `outputs/checkpoints/mlp_classifier.pt`, `outputs/logs/mlp_classifier.json`, `outputs/results/mlp_classifier.json` |
| `python -m train.train_ae` | Trains the Vanilla Autoencoder (latent 64) with an MSE loss and reports test MSE/PSNR/SSIM | `outputs/checkpoints/vanilla_ae.pt`, `outputs/logs/vanilla_ae.json`, `outputs/results/vanilla_ae.json` |
| `python -m train.train_sparse_ae` | Trains the Sparse Autoencoder (256-unit code, MSE + λ‖h‖₁ with λ = 1e-4) and reports test MSE/PSNR/SSIM and code sparsity | `outputs/checkpoints/sparse_ae.pt`, `outputs/logs/sparse_ae.json`, `outputs/results/sparse_ae.json` |
| `python -m train.train_vae` | Trains the VAE (latent 64, negative ELBO = BCE + KL) and reports test MSE/PSNR/SSIM, KL, and active latent dimensions | `outputs/checkpoints/vae.pt`, `outputs/logs/vae.json`, `outputs/results/vae.json` |
| `python -m train.train_gan` | Trains the Vanilla GAN (MLP generator and discriminator, z = 100, non-saturating BCE loss) for 50 epochs | `outputs/checkpoints/vanilla_gan_generator.pt`, `outputs/checkpoints/vanilla_gan_discriminator.pt`, `outputs/logs/vanilla_gan.json` |
| `python -m train.train_dcgan` | Trains the DCGAN (strided-conv discriminator, transposed-conv generator, z = 100) for 50 epochs with the same loop and optimizer settings as the Vanilla GAN | `outputs/checkpoints/dcgan_generator.pt`, `outputs/checkpoints/dcgan_discriminator.pt`, `outputs/logs/dcgan.json` |
| `python -m train.train_ddpm` | Trains the DDPM (small UNet, T = 1000, noise prediction) with an EMA copy of the weights. About 7 minutes per epoch on a CPU (408 s on average in the reported run), so a GPU (e.g. Colab) is advisable for the full run | `outputs/checkpoints/ddpm.pt`, `outputs/logs/ddpm.json` |

Experiments are run the same way and read the trained checkpoints, so run the matching
training script first.

| Command | What it does | Outputs |
|---|---|---|
| `python -m experiments.mlp_class_projection` | Step 2: back-projects one-hot class vectors to pixels with transposed MLP weights (no training) and measures weight orthogonality | `outputs/figures/mlp_class_projection.png`, `outputs/results/mlp_class_projection.json` |
| `python -m experiments.mlp_latent_projection` | Step 3: encodes test images to z_MLP, back-projects with W^T, and scores the result with MSE/PSNR/SSIM against the originals | `outputs/figures/mlp_latent_projection.png`, `outputs/results/mlp_latent_projection.json` |
| `python -m experiments.mlp_frozen_decoding` | Step 4: freezes the MLP classifier, trains a 64→256→784 decoder on z_MLP, and compares it with the Vanilla AE (needs `train_mlp` and `train_ae` first) | `outputs/checkpoints/frozen_latent_decoder.pt`, `outputs/logs/frozen_latent_decoder.json`, `outputs/figures/mlp_frozen_decoding.png`, `outputs/results/mlp_frozen_decoding.json` |
| `python -m experiments.mlp_latent_sampling` | Step 5: fits a diagonal Gaussian to z_MLP on the test set, samples it, decodes with the frozen decoder, and scores realism (needs `experiments.mlp_frozen_decoding` first) | `outputs/figures/mlp_latent_sampling.png`, `outputs/results/mlp_latent_sampling.json` |
| `python -m experiments.compare_models` | Part IV table: FID of 1000 samples per generator against the test set, inference latency per batch of 64, training time and loss stability from the logs, and the autoencoder test metrics (needs all training scripts first). Downloads the Inception weights (about 90 MB) on first use; the DDPM sampling is slow on a CPU but resumes if interrupted | `outputs/results/model_comparison.json`, `outputs/results/model_comparison.csv`, cached samples in `outputs/samples/` (git-ignored) |

Each script is listed here once it is added.

## Project structure

| Path | Description |
|---|---|
| `configs.py` | Single source of seed, paths, device, batch size, and split ratios |
| `data.py` | `set_seed()` and `get_dataloaders()` (merged 70k samples, 80/10/10 split) |
| `utils.py` | `Timer`, JSON helper, `TrainingHistory` (loss and time per epoch), checkpoint save/load, `save_figure` |
| `models/mlp_classifier.py` | Bias-free MLP classifier with an `encode()` method returning the 64-d latent feature |
| `train/train_mlp.py` | Training script for the MLP classifier (keeps the best epoch by validation loss) |
| `experiments/mlp_class_projection.py` | Step 2 of the MLP experiment: class-vector back-projection through W^T and orthogonality check |
| `metrics.py` | `reconstruction_metrics()`: test-set MSE, PSNR (per-image mean), and SSIM for any reconstruction function; `FidScorer`: FID of generated images |
| `experiments/mlp_latent_projection.py` | Step 3 of the MLP experiment: latent transpose projection of real test images |
| `models/vanilla_ae.py` | `VanillaAutoencoder` and the `MLPDecoder` it shares with the frozen-latent experiment |
| `train/engine.py` | Shared training loop (`fit`): best epoch by validation loss, checkpoint, per-epoch loss and time log |
| `train/train_ae.py` | Training script for the Vanilla Autoencoder |
| `experiments/mlp_frozen_decoding.py` | Step 4 of the MLP experiment: frozen-latent decoder versus Vanilla AE |
| `experiments/mlp_latent_sampling.py` | Step 5 of the MLP experiment: Gaussian sampling in the z_MLP space and its decoded images |
| `models/sparse_ae.py` | `SparseAutoencoder`: 784 → 256 ReLU code `h` → 784 |
| `train/train_sparse_ae.py` | Training script for the Sparse Autoencoder (L1 penalty on `h`, sparsity statistics) |
| `models/vae.py` | `VAE`: probabilistic encoder (mu, logvar), reparameterisation, and the shared `MLPDecoder` |
| `train/train_vae.py` | Training script for the VAE (negative ELBO, posterior statistics) |
| `models/gan.py` | `Generator` (100 → 256 → 512 → 1024 → 784) and `Discriminator` (logit output) of the Vanilla GAN |
| `train/gan_engine.py` | Shared adversarial loop (`fit_gan`) for the Vanilla GAN and the DCGAN; logs D/G loss and D(x), D(G(z)) per epoch |
| `train/train_gan.py` | Training script for the Vanilla GAN |
| `models/dcgan.py` | `DCGANGenerator` (transposed convs, BatchNorm) and `DCGANDiscriminator` (strided convs, logit output) |
| `train/train_dcgan.py` | Training script for the DCGAN (uses `train/gan_engine.py`) |
| `models/unet.py` | Small time-conditioned `UNet` (three resolutions, skip connections) that predicts the noise |
| `models/ddpm.py` | `GaussianDiffusion`: linear noise schedule, noise-prediction loss, and ancestral sampling |
| `train/train_ddpm.py` | Training script for the DDPM (EMA weights, fixed-seed validation loss) |
| `models/zoo.py` | Loads trained checkpoints: `load_autoencoder()` and `load_prior_sampler()` (one `sample(n)` interface for VAE, GAN, DCGAN, and DDPM) |
| `experiments/compare_models.py` | Quantitative comparison of the six models: FID, latency, training time and stability, and the Part IV table |
| `requirements.txt` | Pinned dependencies |
| `data/` | Downloaded dataset (git-ignored) |
| `outputs/checkpoints/` | Model weights (git-ignored) |
| `outputs/logs/` | Training logs as JSON |
| `outputs/figures/` | Generated figures |
| `outputs/results/` | Metric tables (CSV/JSON) |
| `outputs/samples/` | Cached generated images used for FID (git-ignored) |

`outputs/` subfolders are created on first write.
