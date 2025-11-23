"""Helper utilities for batching and reproducibility."""
from __future__ import annotations

import numpy as np


def set_seed(seed: int | None = None) -> None:
    """Set numpy RNG seed for reproducibility."""
    if seed is not None:
        np.random.seed(seed)


def batch_indices(n_items: int, batch_size: int):
    """Yield start and end indices for batching."""
    for start in range(0, n_items, batch_size):
        yield start, min(start + batch_size, n_items)


def moving_average(values, window: int = 3):
    """Compute simple moving average for smoothing metrics."""
    if len(values) < window:
        return values
    cumsum = np.cumsum(values, dtype=float)
    return (cumsum[window - 1:] - cumsum[:-window + 1]) / window

