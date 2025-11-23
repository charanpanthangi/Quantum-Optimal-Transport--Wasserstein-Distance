"""Quantum critic circuit approximating Wasserstein critic scores."""
from __future__ import annotations

import numpy as np
import pennylane as qml

from .quantum_embedding import encode_sample_into_circuit

# Use a small default device for fast CPU simulation
critic_dev = qml.device("default.qubit", wires=2, shots=None)


def init_critic_params(n_layers: int = 1, seed: int | None = 0) -> np.ndarray:
    """Initialize critic parameters for a simple layered circuit."""
    rng = np.random.default_rng(seed)
    # Each layer has RX, RY, RZ on both qubits plus a shared CZ
    return rng.uniform(-np.pi, np.pi, size=(n_layers, 6))


@qml.qnode(critic_dev)
def critic_qnode(x: float, params: np.ndarray) -> qml.typing.Result:
    """Return critic score for one sample using expectation of PauliZ."""
    # Encode input sample
    encode_sample_into_circuit(x, qubit=0)
    # Simple data re-uploading on second wire for symmetry
    encode_sample_into_circuit(x, qubit=1)
    for layer in params:
        qml.RX(layer[0], wires=0)
        qml.RY(layer[1], wires=0)
        qml.RZ(layer[2], wires=0)
        qml.RX(layer[3], wires=1)
        qml.RY(layer[4], wires=1)
        qml.RZ(layer[5], wires=1)
        qml.CZ(wires=[0, 1])
    return qml.expval(qml.PauliZ(0))


def critic_forward(x_batch: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Vectorized critic forward pass for a batch of samples."""
    return np.array([critic_qnode(float(x), params) for x in x_batch], dtype=float)

