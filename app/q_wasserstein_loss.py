"""Quantum Wasserstein-style losses using a variational critic."""
from __future__ import annotations

import numpy as np

from .critic_circuit import critic_forward
from .generator_circuit import sample_generator


def critic_loss(params_critic: np.ndarray, params_generator: np.ndarray, x_real_batch: np.ndarray, z_batch: np.ndarray) -> float:
    """Critic loss to maximize: E_real[f] - E_fake[f]."""
    real_scores = critic_forward(x_real_batch, params_critic)
    fake_samples = sample_generator(len(z_batch), params_generator, seed=None)
    fake_scores = critic_forward(fake_samples, params_critic)
    return float(-(np.mean(real_scores) - np.mean(fake_scores)))


def generator_loss(params_critic: np.ndarray, params_generator: np.ndarray, z_batch: np.ndarray) -> float:
    """Generator loss to minimize: -E_fake[f]."""
    fake_samples = sample_generator(len(z_batch), params_generator, seed=None)
    fake_scores = critic_forward(fake_samples, params_critic)
    return float(-np.mean(fake_scores))


def estimate_wasserstein_distance(params_critic: np.ndarray, x_real: np.ndarray, x_fake: np.ndarray) -> float:
    """Quantum Wasserstein estimate based on critic expectations."""
    real_scores = critic_forward(x_real, params_critic)
    fake_scores = critic_forward(x_fake, params_critic)
    return float(np.mean(real_scores) - np.mean(fake_scores))

