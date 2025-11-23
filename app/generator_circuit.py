"""Quantum generator circuit producing simple 1D samples."""
from __future__ import annotations

import numpy as np
import pennylane as qml

# Single-qubit generator for speed
_gen_dev = qml.device("default.qubit", wires=1, shots=None)


def init_generator_params(n_layers: int = 1, seed: int | None = 0) -> np.ndarray:
    """Initialize generator parameters for RX/RY/RZ layers."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-np.pi, np.pi, size=(n_layers, 3))


@qml.qnode(_gen_dev)
def _generator_qnode(z: float, params: np.ndarray) -> qml.typing.Result:
    """Return probability of measuring |1> given latent input and params."""
    # Encode latent variable as a rotation about Y for smooth control
    qml.RY(qml.math.pi * z, wires=0)
    for layer in params:
        qml.RX(layer[0], wires=0)
        qml.RY(layer[1], wires=0)
        qml.RZ(layer[2], wires=0)
    return qml.probs(wires=0)


def sample_generator(n_samples: int, params: np.ndarray, seed: int | None = None) -> np.ndarray:
    """Generate approximate continuous samples in [-1, 1] from the PQC.

    The probability of outcome |1> is mapped to a real value via x = 2*p1 - 1.
    Args:
        n_samples: Number of generated points.
        params: Generator circuit parameters.
        seed: Optional seed for latent sampling.

    Returns:
        Array of generated samples in [-1, 1].
    """
    rng = np.random.default_rng(seed)
    z_vals = rng.random(n_samples)
    outputs = []
    for z in z_vals:
        probs = _generator_qnode(float(z), params)
        p1 = float(probs[1])
        outputs.append(2 * p1 - 1)
    return np.array(outputs, dtype=float)

