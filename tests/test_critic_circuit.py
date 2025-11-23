import numpy as np

from app.critic_circuit import critic_forward, init_critic_params


def test_critic_forward_shape():
    params = init_critic_params(n_layers=1, seed=0)
    x = np.linspace(-1, 1, 5)
    scores = critic_forward(x, params)
    assert scores.shape == (5,)
    assert np.all(np.isfinite(scores))
