import pennylane as qml

from app.quantum_embedding import encode_sample_into_circuit


def test_encoding_runs():
    dev = qml.device("default.qubit", wires=1, shots=None)

    @qml.qnode(dev)
    def circuit(x):
        encode_sample_into_circuit(x, qubit=0)
        return qml.state()

    out = circuit(0.2)
    assert out.shape[0] == 2
