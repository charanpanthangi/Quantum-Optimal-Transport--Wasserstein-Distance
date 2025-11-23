import numpy as np

from app.generator_circuit import init_generator_params, sample_generator


def test_generator_outputs_range():
    params = init_generator_params(n_layers=1, seed=0)
    samples = sample_generator(10, params, seed=1)
    assert samples.shape == (10,)
    assert np.all(samples >= -1.0) and np.all(samples <= 1.0)
    assert np.all(np.isfinite(samples))
