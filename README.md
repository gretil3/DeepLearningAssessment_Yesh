# Comparative Study of Generative Models (Deep Learning D6421, Session 07)

A PyTorch study that compares six generative models on Fashion-MNIST: Vanilla
Autoencoder, Sparse Autoencoder, VAE, Vanilla GAN, DCGAN, and DDPM. An MLP
classifier is also trained to show why discriminative features cannot be used
directly for image generation.

All models share the same data split, batch size, and random seed (42), and every
metric is computed on the held-out test set.

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
| `python -m experiments.sparse_lambda_sweep` | Question Q1: trains the Sparse Autoencoder for each λ in {0, 1e-5, 1e-4, 1e-3} with identical seed, data order, and epochs, and reports test MSE/PSNR/SSIM and code sparsity | `outputs/logs/sparse_ae_lambda_<λ>.json`, `outputs/results/sparse_lambda_sweep.json`, `outputs/results/sparse_lambda_sweep.csv` |
| `python -m experiments.compare_models` | Part IV table: FID of 1000 samples per generator against the test set, inference latency per batch of 64, training time and loss stability from the logs, and the autoencoder test metrics (needs all training scripts first). Downloads the Inception weights (about 90 MB) on first use; the DDPM sampling is slow on a CPU but resumes if interrupted | `outputs/results/model_comparison.json`, `outputs/results/model_comparison.csv`, cached samples in `outputs/samples/` (git-ignored) |
| `python -m experiments.make_figures` | Part V: the five comparison figures (reconstruction grid, random sampling, MLP-limitation evidence, latent interpolation, loss curves with the DDPM denoising trajectory). Needs every training script, `experiments.mlp_frozen_decoding`, and `experiments.compare_models` (its cached samples feed Figure 2) | `outputs/figures/fig1_reconstruction_grid.png`, `fig2_random_sampling.png`, `fig3_mlp_limitation_evidence.png`, `fig4_latent_interpolation.png`, `fig5_training_curves_and_ddpm_trajectory.png` |

## Reproducing the results

Model weights are git-ignored, so a fresh clone must train the models before running the
experiments. The logs, result tables, and figures that this run produced are committed in
`outputs/`. Run the commands from the repository root, in this order:

```bash
# 1. MLP classifier and autoencoders
python -m train.train_mlp
python -m train.train_ae
python -m train.train_sparse_ae
python -m train.train_vae

# 2. Adversarial and diffusion models
python -m train.train_gan
python -m train.train_dcgan
python -m train.train_ddpm

# 3. Why a classifier cannot generate images (Steps 2-5 of the MLP experiment)
python -m experiments.mlp_class_projection
python -m experiments.mlp_latent_projection
python -m experiments.mlp_frozen_decoding
python -m experiments.mlp_latent_sampling

# 4. Sparsity sweep, model comparison, and report figures
python -m experiments.sparse_lambda_sweep
python -m experiments.compare_models
python -m experiments.make_figures
```

The code selects CUDA when it is available and falls back to the CPU, so the same
commands run on a local machine or on Colab (`pip install -r requirements.txt` first).
The reported run used a CPU only. The DDPM is the slow part: about 7 minutes per epoch
on that CPU, and 1000 sampling steps per image, which makes `compare_models` take about an
hour there. All hyperparameters live in `configs.py`.

### Reproducibility

- Seed 42 for `torch`, `numpy`, and `random` (`data.set_seed()`).
- The 70,000 Fashion-MNIST images (train and test merged) are split 80/10/10 into
  56,000 / 7,000 / 7,000 with a dedicated `torch.Generator` seeded with 42, so every model
  sees the same split.
- `batch_size = 128`; shuffling is on for the training split only.
- All reported metrics are computed on the test split.
- Rerunning a script with the same seed reproduces its numbers on the same machine.
  Training time and latency depend on the hardware.

## Results

All numbers come from one training run per model (seed 42) on the test split. They are
produced by `experiments.compare_models` and stored in `outputs/results/model_comparison.csv`.

| Model | Latent dim | Train time / epoch (s) | Latency (ms / batch of 64) | Test MSE | Test PSNR (dB) | Test SSIM | FID |
|---|---|---|---|---|---|---|---|
| Vanilla Autoencoder | 64 | 11.4 | 0.33 (reconstruction) | 0.00702 | 22.61 | 0.855 | N/A |
| Sparse Autoencoder | 256 (sparse code) | 9.4 | 0.24 (reconstruction) | 0.00693 | 22.52 | 0.798 | N/A |
| VAE | 64 | 7.7 | 0.32 (generation) | 0.01367 | 19.48 | 0.761 | 100.7 |
| Vanilla GAN | 100 (prior noise) | 27.9 | 1.26 (generation) | N/A | N/A | N/A | 123.0 |
| DCGAN | 100 (prior noise) | 71.7 | 7.56 (generation) | N/A | N/A | N/A | 34.5 |
| DDPM | 28×28 (image space) | 407.9 | 121,636 (generation, 1000 steps) | N/A | N/A | N/A | 35.8 |

How to read the table:

- FID uses 1000 generated images per model against the 7,000 test images. Two real sets of
  that size already score 16.2 (`fid_real_vs_real_baseline` in `model_comparison.json`), which
  is the practical floor. The Inception network sees 28×28 grey images enlarged to 299×299,
  so the values are only comparable between the models in this repository, not with
  published FIDs.
- The DCGAN and DDPM scores differ by about 1.2 on a single seed and 1000 samples, which is
  too small to rank them. Both are far better than the VAE and the Vanilla GAN.
- The autoencoders have no generative FID (sampling from a prior is not part of their design);
  GANs and the DDPM have no encoder, so reconstruction metrics do not apply.
- Latency was measured on the CPU, 20 repeats (3 warm-up) per model; the DDPM was timed with
  a single 1000-step run. Autoencoder latency is a reconstruction pass, generator latency a
  sampling pass from the prior.
- Per-epoch losses, their final values, and how settled they are over the last 10 epochs are
  in `outputs/results/model_comparison.json` under `training`.

### Sparsity weight sweep

The same 256-unit Sparse Autoencoder trained with different L1 weights λ (same seed, data
order, and 20 epochs; `experiments.sparse_lambda_sweep`, `outputs/results/sparse_lambda_sweep.csv`).
λ = 0 is the unpenalised baseline and λ = 1e-4 is the Sparse Autoencoder used everywhere else.

| λ | Test MSE | Test PSNR (dB) | Test SSIM | Mean ‖h‖₁ | Zero units | Active units (of 256) |
|---|---|---|---|---|---|---|
| 0 | 0.00395 | 25.01 | 0.888 | 374.5 | 24.5% | 193.3 |
| 1e-5 | 0.00408 | 24.94 | 0.877 | 104.7 | 32.5% | 172.8 |
| **1e-4** | **0.00693** | **22.52** | **0.798** | **24.7** | **68.4%** | **81.0** |
| 1e-3 | 0.03945 | 14.61 | 0.402 | 2.9 | 98.9% | 2.7 |

A larger λ shrinks the code (‖h‖₁ falls by more than an order of magnitude from λ = 0 to
1e-4) and silences more units, at the price of reconstruction fidelity. Between 1e-5 and
1e-4 the penalty already costs 0.0028 MSE, and at 1e-3 almost every unit is dead and the
images are no longer recognisable. The sparse model at λ = 1e-4 is therefore a trade of
fidelity for sparsity, not a free improvement over the unpenalised code.

### MLP classifier experiment

The 784→256→64→10 bias-free classifier reaches 88.56% test accuracy (target ≥ 88%). Its
weights and features are then used as a decoder in four ways (`outputs/results/mlp_*.json`):

| Step | Test result |
|---|---|
| 2. Back-project one-hot class vectors with Wᵀ | Mean correlation with the class-mean image 0.18; no recognisable objects |
| 3. Back-project z_MLP of test images with Wᵀ | MSE 0.287, PSNR 5.47 dB, SSIM 0.059; worse than the constant mean image (MSE 0.087) |
| 4. Train a decoder on frozen z_MLP | MSE 0.0167, PSNR 18.54 dB, SSIM 0.716, versus the Vanilla AE at 0.00702, 22.61 dB, 0.855 (2.4× the MSE) |
| 5. Sample z ~ N(μ_z, diag σ_z²) and decode | Distance to the nearest real image 0.059 versus 0.024 for real images; 48% of real z_MLP entries are exactly zero while 19% of the Gaussian samples are negative |

## Figures

The five report figures are generated by `experiments.make_figures` (300 dpi).

| Figure | File |
|---|---|
| 1. Reconstruction grid: original, Vanilla AE, Sparse AE, VAE | `outputs/figures/fig1_reconstruction_grid.png` |
| 2. Random samples (4×4) of VAE, Vanilla GAN, DCGAN, DDPM | `outputs/figures/fig2_random_sampling.png` |
| 3. Evidence of the MLP classifier's limits | `outputs/figures/fig3_mlp_limitation_evidence.png` |
| 4. Linear latent interpolation, Trouser → Sneaker | `outputs/figures/fig4_latent_interpolation.png` |
| 5. Training loss curves and the DDPM reverse trajectory | `outputs/figures/fig5_training_curves_and_ddpm_trajectory.png` |

In Figure 4 the autoencoders interpolate between the codes of two real test images (the VAE
between their posterior means). The GANs have no encoder, so they interpolate between two
random latents that decode to a Trouser and a Sneaker according to the MLP classifier.

## Limitations

- One training run (seed 42) per model; no confidence intervals. Differences that are small
  relative to run-to-run variation, such as DCGAN versus DDPM FID, should not be read as a
  ranking.
- The autoencoders are trained for 20 epochs and their validation loss was still falling
  slowly.
- The DDPM is trained for 15 epochs only (sized for a CPU run), so its samples are rougher
  than a longer or GPU-trained run would give.
- The VAE uses 16 of its 64 latent dimensions; the rest collapse to the prior, which limits
  its sharpness.
- Latency is a CPU measurement; the relative ordering, not the absolute numbers, transfers to
  a GPU.

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
| `models/ddpm.py` | `GaussianDiffusion`: linear noise schedule, noise-prediction loss, ancestral sampling, and `sample_trajectory()` for the denoising figure |
| `train/train_ddpm.py` | Training script for the DDPM (EMA weights, fixed-seed validation loss) |
| `models/zoo.py` | Loads trained checkpoints: `load_autoencoder()` and `load_prior_sampler()` (one `sample(n)` interface for VAE, GAN, DCGAN, and DDPM) |
| `experiments/compare_models.py` | Quantitative comparison of the six models: FID, latency, training time and stability, and the Part IV table |
| `experiments/sparse_lambda_sweep.py` | Sparse Autoencoder trained for several L1 weights λ (question Q1) |
| `experiments/make_figures.py` | The five report figures of Part V |
| `requirements.txt` | Pinned dependencies |
| `data/` | Downloaded dataset (git-ignored) |
| `outputs/checkpoints/` | Model weights (git-ignored) |
| `outputs/logs/` | Training logs as JSON |
| `outputs/figures/` | Generated figures |
| `outputs/results/` | Metric tables (CSV/JSON) |
| `outputs/samples/` | Cached generated images used for FID (git-ignored) |

`outputs/` subfolders are created on first write.
