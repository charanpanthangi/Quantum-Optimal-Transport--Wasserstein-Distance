"""Utilities for encoding 1D samples into quantum circuits."""
from __future__ import annotations

import pennylane as qml


def angle_for_sample(x: float) -> float:
    """Map x in [-1, 1] to a rotation angle in [0, pi]."""
    return qml.math.pi * (x + 1) / 2


def encode_sample_into_circuit(x: float, qubit: int = 0) -> None:
    """Encode a single sample into the given qubit using an RY rotation.

    Args:
        x: Sample value in [-1, 1].
        qubit: Target qubit index; defaults to 0 for single-qubit circuits.
    """
    theta = angle_for_sample(x)
    qml.RY(theta, wires=qubit)

