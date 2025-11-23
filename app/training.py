"""Training loop for a tiny quantum Wasserstein GAN-style model."""
from __future__ import annotations

import numpy as np

from .classical_wasserstein import wasserstein_1d
from .critic_circuit import init_critic_params
from .generator_circuit import init_generator_params, sample_generator
from .q_wasserstein_loss import critic_loss, generator_loss, estimate_wasserstein_distance
from .utils import batch_indices, set_seed


def train_qot(
    n_epochs: int = 20,
    batch_size: int = 32,
    n_critic_steps: int = 2,
    critic_layers: int = 1,
    generator_layers: int = 1,
    seed: int | None = 0,
):
    """Run a minimal training loop returning parameters and history."""
    set_seed(seed)
    params_critic = init_critic_params(n_layers=critic_layers, seed=seed)
    params_generator = init_generator_params(
        n_layers=generator_layers, seed=seed + 1 if seed is not None else None
    )

    from .dataset import make_dataset

    x_train, _ = make_dataset(n_train=256, n_val=128, seed=seed)
    history = {
        "classical_w1": [],
        "quantum_estimate": [],
        "gen_loss": [],
        "critic_loss": [],
    }

    lr = 0.05
    for epoch in range(n_epochs):
        rng = np.random.default_rng(seed + epoch if seed is not None else None)
        perm = rng.permutation(len(x_train))
        x_train = x_train[perm]

        for start, end in batch_indices(len(x_train), batch_size):
            x_real = x_train[start:end]
            # Critic updates (simple sign gradient descent on negative objective)
            for _ in range(n_critic_steps):
                z_batch = rng.random(len(x_real))
                loss_c = critic_loss(params_critic, params_generator, x_real, z_batch)
                params_critic = params_critic - lr * np.sign(loss_c)

            # Generator update
            z_batch = rng.random(len(x_real))
            loss_g = generator_loss(params_critic, params_generator, z_batch)
            params_generator = params_generator - lr * np.sign(loss_g)

        # Logging
        x_fake_full = sample_generator(len(x_train), params_generator, seed=seed)
        class_w1 = wasserstein_1d(x_train, x_fake_full)
        q_est = estimate_wasserstein_distance(
            params_critic, x_train[: batch_size], x_fake_full[: batch_size]
        )
        history["classical_w1"].append(class_w1)
        history["quantum_estimate"].append(q_est)
        history["gen_loss"].append(loss_g)
        history["critic_loss"].append(loss_c)

    return params_critic, params_generator, history

