"""Command-line entry point for the quantum optimal transport demo."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .dataset import make_dataset
from .generator_circuit import sample_generator
from .plots import (
    plot_data_vs_model_histograms,
    plot_generator_samples_evolution,
    plot_quantum_vs_classical_final,
    plot_wasserstein_convergence,
)
from .training import train_qot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantum Wasserstein distance demo")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--critic-steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples_dir = Path("examples")
    examples_dir.mkdir(parents=True, exist_ok=True)

    x_train, _ = make_dataset(n_train=256, n_val=128, seed=args.seed)
    params_c, params_g, history = train_qot(
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        n_critic_steps=args.critic_steps,
        seed=args.seed,
    )

    # Sample final generator outputs
    x_fake = sample_generator(len(x_train), params_g, seed=args.seed + 42)

    # Collect simple checkpoints for illustration
    checkpoints = {"start": sample_generator(128, params_g, seed=args.seed)}
    checkpoints["final"] = x_fake

    plot_data_vs_model_histograms(
        x_train, x_fake, output_path=examples_dir / "qot_data_vs_model_histograms.svg"
    )
    plot_wasserstein_convergence(
        history, output_path=examples_dir / "qot_wasserstein_convergence.svg"
    )
    plot_quantum_vs_classical_final(
        history, output_path=examples_dir / "qot_quantum_vs_classical_wasserstein.svg"
    )
    plot_generator_samples_evolution(
        checkpoints, output_path=examples_dir / "qot_generator_samples_evolution.svg"
    )

    print("Training complete.")
    print("Final classical W1:", history["classical_w1"][-1])
    print("Final quantum estimate:", history["quantum_estimate"][-1])
    print("SVG plots saved to examples/ directory.")


if __name__ == "__main__":
    main()

