import numpy as np

from app.dataset import make_dataset, sample_data


def test_sample_range():
    samples = sample_data(50, seed=0)
    assert samples.shape == (50,)
    assert np.all(samples >= -1.0) and np.all(samples <= 1.0)


def test_make_dataset_split():
    train, val = make_dataset(32, 16, seed=1)
    assert train.shape == (32,)
    assert val.shape == (16,)
