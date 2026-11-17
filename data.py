"""Fashion-MNIST data pipeline: seeding, merging, splitting, and dataloaders."""

import random

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, random_split
from torchvision import datasets, transforms

import configs


def set_seed(seed: int = configs.SEED) -> None:
    """Seed every RNG used in the project for reproducible runs.

    Args:
        seed: Seed value shared by random, numpy, torch, and cuda.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Safe no-op when CUDA is unavailable.
    torch.cuda.manual_seed_all(seed)


def get_dataloaders() -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test loaders from the merged Fashion-MNIST train+test sets.

    The official 60k train and 10k test sets are merged (70k) and re-split with a
    fixed generator, so every script sees the identical 80/10/10 partition.

    Returns:
        A (train_loader, val_loader, test_loader) tuple. Only the train loader shuffles.
    """
    # Pixels stay in [0, 1]; models needing another range rescale on their own.
    to_tensor = transforms.ToTensor()
    train_set = datasets.FashionMNIST(
        configs.DATA_DIR, train=True, download=True, transform=to_tensor
    )
    test_set = datasets.FashionMNIST(
        configs.DATA_DIR, train=False, download=True, transform=to_tensor
    )
    full_set = ConcatDataset([train_set, test_set])

    n_total = len(full_set)
    n_train = round(configs.SPLIT_RATIOS[0] * n_total)
    n_val = round(configs.SPLIT_RATIOS[1] * n_total)
    # The remainder goes to test so the three sizes always sum to n_total.
    n_test = n_total - n_train - n_val

    # Dedicated generator: the split must not depend on global RNG state.
    split_generator = torch.Generator().manual_seed(configs.SEED)
    train_ds, val_ds, test_ds = random_split(
        full_set, [n_train, n_val, n_test], generator=split_generator
    )

    # Same reasoning: a private generator keeps the shuffle order reproducible.
    shuffle_generator = torch.Generator().manual_seed(configs.SEED)
    loader_kwargs = {
        "batch_size": configs.BATCH_SIZE,
        "num_workers": configs.NUM_WORKERS,
        "pin_memory": configs.DEVICE.type == "cuda",
    }
    train_loader = DataLoader(
        train_ds, shuffle=True, generator=shuffle_generator, **loader_kwargs
    )
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader
