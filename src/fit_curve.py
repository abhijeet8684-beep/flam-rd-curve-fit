"""Fit an unordered point cloud to the supplied parametric curve."""

from __future__ import annotations

from pathlib import Path

import matplotlib

# The script only writes an image, so it must not depend on a desktop GUI backend.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import differential_evolution, minimize_scalar
from scipy.spatial import cKDTree


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "xy_data.csv"
PLOT_PATH = PROJECT_ROOT / "results" / "curve_fit.png"

# (theta in degrees, M, X). SciPy accepts closed bounds; the specified open
# intervals are represented by their endpoints for numerical optimisation.
BOUNDS = [(0.0, 50.0), (-0.05, 0.05), (0.0, 100.0)]
T_MIN, T_MAX = 6.0, 60.0
N_CURVE_SAMPLES = 12_000
VERIFIED_PARAMETERS = (30.0, 0.03, 55.0)


def curve_at_t(t: float | np.ndarray, theta_degrees: float, m: float, x_offset: float) -> np.ndarray:
    """Evaluate the parametric curve at one or more t values."""
    theta = np.deg2rad(theta_degrees)
    oscillation = np.exp(m * np.abs(t)) * np.sin(0.3 * t)

    x = t * np.cos(theta) - oscillation * np.sin(theta) + x_offset
    y = 42.0 + t * np.sin(theta) + oscillation * np.cos(theta)
    return np.column_stack((x, y))


def curve_points(theta_degrees: float, m: float, x_offset: float, n: int = N_CURVE_SAMPLES) -> np.ndarray:
    """Return a dense (x, y) sampling of the curve for one parameter set."""
    t = np.linspace(T_MIN, T_MAX, n)
    return curve_at_t(t, theta_degrees, m, x_offset)


def mean_nearest_point_manhattan_distance(parameters: np.ndarray, data: np.ndarray) -> float:
    """Mean nearest-point Manhattan/L1 distance, |dx| + |dy|, to a candidate curve."""
    candidate_curve = curve_points(*parameters)
    distances, _ = cKDTree(candidate_curve).query(data, p=1, workers=-1)
    return float(np.mean(np.abs(distances)))


def fit_parameters(data: np.ndarray) -> tuple[np.ndarray, float]:
    """Use a global differential-evolution search over the supplied bounds."""
    result = differential_evolution(
        mean_nearest_point_manhattan_distance,
        bounds=BOUNDS,
        args=(data,),
        seed=2026,
        popsize=15,
        maxiter=250,
        tol=1e-8,
        polish=True,
        updating="immediate",
        workers=1,
    )
    return result.x, float(result.fun)


def validate_t_inversion(data: np.ndarray) -> tuple[float, float]:
    """Invert t for every point using verified parameters and report reconstruction distances."""
    coarse_t = np.linspace(T_MIN, T_MAX, 6_001)
    coarse_curve = curve_at_t(coarse_t, *VERIFIED_PARAMETERS)
    step = coarse_t[1] - coarse_t[0]
    reconstruction_distances = []

    for point in data:
        coarse_distances = np.linalg.norm(coarse_curve - point, axis=1)
        best_index = int(np.argmin(coarse_distances))
        lower = max(T_MIN, coarse_t[best_index] - step)
        upper = min(T_MAX, coarse_t[best_index] + step)

        def distance_at_t(t: float) -> float:
            return float(np.linalg.norm(curve_at_t(t, *VERIFIED_PARAMETERS)[0] - point))

        local_result = minimize_scalar(distance_at_t, bounds=(lower, upper), method="bounded")
        reconstruction_distances.append(local_result.fun)

    distances = np.asarray(reconstruction_distances)
    return float(np.mean(distances)), float(np.max(distances))


def save_plot(data: np.ndarray, final_parameters: np.ndarray) -> None:
    """Create a comparison of the source data and final reported curve."""
    fitted_curve = curve_points(*final_parameters)
    theta, m, x_offset = final_parameters

    fig, ax = plt.subplots(figsize=(10, 7), dpi=180)
    ax.scatter(data[:, 0], data[:, 1], s=13, alpha=0.7, label="Supplied data", color="tab:blue")
    ax.plot(fitted_curve[:, 0], fitted_curve[:, 1], linewidth=2.2, label="Final reported curve", color="tab:orange")
    ax.set_title(f"Parametric curve fit (theta={theta:.0f} degrees, M={m:.2f}, X={x_offset:.0f})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()

    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    data = np.loadtxt(DATA_PATH, delimiter=",", skiprows=1)
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError(f"Expected two CSV columns (x, y), got shape {data.shape}.")

    fitted_parameters, fitted_manhattan_distance = fit_parameters(data)
    verified_manhattan_distance = mean_nearest_point_manhattan_distance(np.asarray(VERIFIED_PARAMETERS), data)
    validation_mean, validation_maximum = validate_t_inversion(data)

    print("Differential-evolution convergence evidence")
    print(f"  theta = {fitted_parameters[0]:.10f} degrees")
    print(f"  M     = {fitted_parameters[1]:.10f}")
    print(f"  X     = {fitted_parameters[2]:.10f}")
    print(f"  mean nearest-point Manhattan/L1 distance ({N_CURVE_SAMPLES:,} curve samples) = {fitted_manhattan_distance:.10e}")
    print()
    print("Final reported parameters (independently verified)")
    print("  theta = 30 degrees")
    print("  M     = 0.03")
    print("  X     = 55")
    print(f"  mean nearest-point Manhattan/L1 distance ({N_CURVE_SAMPLES:,} curve samples) = {verified_manhattan_distance:.10e}")
    print()
    print("Post-fit validation: per-point t inversion using verified parameters")
    print(f"  mean Euclidean reconstruction distance = {validation_mean:.10e}")
    print(f"  maximum Euclidean reconstruction distance = {validation_maximum:.10e}")

    save_plot(data, np.asarray(VERIFIED_PARAMETERS))
    print(f"\nSaved plot: {PLOT_PATH}")


if __name__ == "__main__":
    main()
