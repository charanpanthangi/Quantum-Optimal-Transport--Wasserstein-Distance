import numpy as np

from app.classical_wasserstein import wasserstein_1d


def test_non_negative_and_zero_self():
    x = np.linspace(-1, 1, 10)
    y = x.copy()
    assert wasserstein_1d(x, y) >= 0
    assert wasserstein_1d(x, y) < 1e-6

