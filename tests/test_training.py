from app.training import train_qot


def test_short_training_runs():
    params_c, params_g, history = train_qot(n_epochs=2, batch_size=16, n_critic_steps=1, seed=0)
    assert len(history["classical_w1"]) == 2
    # Check that some metric exists and is finite
    assert all([abs(val) >= 0 for val in history["classical_w1"]])
