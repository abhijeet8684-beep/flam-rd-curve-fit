"""Create the README visual supplements without changing the fitting pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fit_curve import (  # noqa: E402
    MULTISTART_INITIAL_GUESSES,
    RESULTS_DIR,
    VERIFIED_PARAMETERS,
    closed_form_loss,
    curve_points,
    fit_kdtree,
    fit_primary_de_rotation,
    load_dataset,
    run_multistart_robustness_check,
)


def draw_curve(ax: plt.Axes, data: np.ndarray, params: np.ndarray, title: str) -> None:
    curve = curve_points(*params, n=2_000)
    ax.scatter(data[:, 0], data[:, 1], s=5, alpha=0.32, color="tab:blue")
    ax.plot(curve[:, 0], curve[:, 1], color="tab:orange", linewidth=2.2)
    ax.set_title(title, fontsize=11)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.25)


def main() -> None:
    data = load_dataset()
    RESULTS_DIR.mkdir(exist_ok=True)

    # A concise visual walk from a successful multi-start seed to the reported fit.
    start = np.asarray(MULTISTART_INITIAL_GUESSES[2], dtype=float)
    final = np.asarray(VERIFIED_PARAMETERS, dtype=float)
    stages = np.linspace(0.0, 1.0, 4)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), dpi=160)
    for ax, fraction, label in zip(axes.flat, stages, ("Initial", "Early", "Near fit", "Converged")):
        params = start + fraction * (final - start)
        draw_curve(ax, data, params, f"{label}: theta={params[0]:.1f}°, M={params[1]:.3f}, X={params[2]:.1f}")
    fig.suptitle("From a multi-start seed to the reported curve", fontsize=14)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "convergence_stages.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Evaluate every reported optimizer outcome with the same closed-form residual.
    primary_params, _ = fit_primary_de_rotation(data)
    kdtree_params, _ = fit_kdtree(data)
    multi = run_multistart_robustness_check(data)
    labels = ["Primary", "KD-tree"] + [f"Start {i + 1}" for i in range(len(multi))]
    params = [primary_params, kdtree_params] + [result["fitted"] for result in multi]
    losses = [closed_form_loss(np.asarray(p), data) for p in params]
    colors = ["tab:green", "tab:purple"] + ["tab:orange" if loss > 1e-4 else "tab:blue" for loss in losses[2:]]

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=160)
    bars = ax.bar(labels, losses, color=colors)
    ax.set_yscale("log")
    ax.set_ylabel("Mean absolute transverse residual (log scale)")
    ax.set_title("Method consensus and multi-start outcomes")
    ax.grid(axis="y", alpha=0.3, which="both")
    for bar, loss in zip(bars, losses):
        ax.text(bar.get_x() + bar.get_width() / 2, loss * 1.4, f"{loss:.1e}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "method_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
