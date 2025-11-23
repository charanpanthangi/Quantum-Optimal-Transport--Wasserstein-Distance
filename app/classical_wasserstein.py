"""Classical 1D Wasserstein-1 distance for empirical samples."""
from __future__ import annotations

import numpy as np


def wasserstein_1d(x_data: np.ndarray, x_model: np.ndarray) -> float:
    """Approximate the Wasserstein-1 distance between two 1D sample sets.

    We sort both arrays and average absolute differences. This is valid for
    empirical 1D distributions and is cheap to compute.

    Args:
        x_data: Samples from the target distribution.
        x_model: Samples from the model distribution.

    Returns:
        Estimated W1 distance as a float.
    """
    if x_data.size == 0 or x_model.size == 0:
        return float("nan")
    n = min(len(x_data), len(x_model))
    data_sorted = np.sort(x_data)[:n]
    model_sorted = np.sort(x_model)[:n]
    return float(np.mean(np.abs(data_sorted - model_sorted)))

