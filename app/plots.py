"""SVG plotting utilities for the quantum optimal transport demo."""
from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns

# Ensure SVG backend
plt.switch_backend("Agg")


def _setup_svg():
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.format"] = "svg"
    plt.rcParams["svg.fonttype"] = "none"


def plot_data_vs_model_histograms(x_real, x_fake, output_path: str) -> None:
    """Overlay histograms of real and generated samples."""
    _setup_svg()
    plt.figure(figsize=(6, 4))
    sns.histplot(x_real, color="tab:blue", alpha=0.5, label="data", bins=20, stat="density")
    sns.histplot(x_fake, color="tab:orange", alpha=0.5, label="generator", bins=20, stat="density")
    plt.xlabel("x")
    plt.ylabel("density")
    plt.title("Toy 1D data vs quantum generator")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, format="svg")
    plt.close()


def plot_wasserstein_convergence(history: dict, output_path: str) -> None:
    """Plot classical and quantum Wasserstein estimates over epochs."""
    _setup_svg()
    plt.figure(figsize=(6, 4))
    plt.plot(history.get("classical_w1", []), label="classical W1")
    plt.plot(history.get("quantum_estimate", []), label="quantum estimate")
    plt.xlabel("epoch")
    plt.ylabel("distance")
    plt.title("Wasserstein distance over training")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, format="svg")
    plt.close()


def plot_quantum_vs_classical_final(history: dict, output_path: str) -> None:
    """Bar plot comparing final classical and quantum distances."""
    _setup_svg()
    plt.figure(figsize=(5, 4))
    labels = ["classical", "quantum"]
    vals = [history.get("classical_w1", [0])[-1], history.get("quantum_estimate", [0])[-1]]
    sns.barplot(x=labels, y=vals, palette=["tab:blue", "tab:orange"])
    plt.ylabel("distance")
    plt.title("Final Wasserstein comparison")
    plt.tight_layout()
    plt.savefig(output_path, format="svg")
    plt.close()


def plot_generator_samples_evolution(samples_per_checkpoint, output_path: str) -> None:
    """Plot generator histograms across checkpoints to show learning."""
    _setup_svg()
    n_steps = len(samples_per_checkpoint)
    fig, axes = plt.subplots(1, n_steps, figsize=(4 * n_steps, 3), sharey=True)
    if n_steps == 1:
        axes = [axes]
    for idx, (step, samples) in enumerate(samples_per_checkpoint.items()):
        sns.histplot(samples, bins=15, color="tab:green", alpha=0.7, ax=axes[idx], stat="density")
        axes[idx].set_title(f"step {step}")
        axes[idx].set_xlabel("x")
    axes[0].set_ylabel("density")
    fig.suptitle("Generator samples through training", y=1.05)
    plt.tight_layout()
    plt.savefig(output_path, format="svg")
    plt.close()

