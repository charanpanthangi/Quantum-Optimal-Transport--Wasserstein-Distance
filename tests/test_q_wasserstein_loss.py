import numpy as np

from app.critic_circuit import init_critic_params
from app.generator_circuit import init_generator_params
from app.q_wasserstein_loss import critic_loss, generator_loss


def test_losses_finite():
    params_c = init_critic_params(1, seed=0)
    params_g = init_generator_params(1, seed=1)
    x_real = np.linspace(-0.5, 0.5, 4)
    z_batch = np.linspace(0, 1, 4)
    lc = critic_loss(params_c, params_g, x_real, z_batch)
    lg = generator_loss(params_c, params_g, z_batch)
    assert np.isfinite(lc)
    assert np.isfinite(lg)
