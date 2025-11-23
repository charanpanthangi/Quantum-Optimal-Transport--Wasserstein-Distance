"""Dataset utilities for toy 1D target distribution."""
from __future__ import annotations

import numpy as np


def sample_data(n_samples: int, seed: int | None = None) -> np.ndarray:
    """Sample from a simple bimodal 1D distribution clipped to [-1, 1].

    Args:
        n_samples: Number of samples to draw.
        seed: Optional random seed for reproducibility.

    Returns:
        Array of shape (n_samples,) with values in [-1, 1].
    """
    rng = np.random.default_rng(seed)
    # Choose component: 0 -> left peak, 1 -> right peak
    components = rng.integers(0, 2, size=n_samples)
    means = np.where(components == 0, -0.5, 0.5)
    std = 0.1
    samples = rng.normal(loc=means, scale=std)
    # Clip to stay in the encoding-friendly range
    return np.clip(samples, -1.0, 1.0)


def make_dataset(n_train: int = 256, n_val: int = 256, seed: int | None = 0) -> tuple[np.ndarray, np.ndarray]:
    """Create train/validation splits from the toy distribution.

    Args:
        n_train: Number of training samples.
        n_val: Number of validation samples.
        seed: Seed for reproducibility; train/val use different seeds for variety.

    Returns:
        Tuple of (train_samples, val_samples).
    """
    train = sample_data(n_train, seed=seed)
    val = sample_data(n_val, seed=seed + 1 if seed is not None else None)
    return train, val

