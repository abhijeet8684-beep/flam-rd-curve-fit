"""Verification script for README approach claims.

Checks two things:
1. Naive row-order regression: assigns CSV row index as evenly-spaced t and
   fits the data directly. Reports actual R².
2. KD-tree alone (no closed-form): runs fit_kdtree() in isolation and
   reports the actual convergence loss from the original method.

Run: python scratch/verify_claims.py
"""
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fit_curve import (
    DATA_PATH, BOUNDS, T_MIN, T_MAX, N_CURVE_SAMPLES,
    VERIFIED_PARAMETERS, curve_at_t, curve_points,
    load_dataset, fit_kdtree,
)


# ─── Claim 1: Naive row-order regression ────────────────────────────────────

def naive_row_order_fit():
    """Assign evenly-spaced t values in row order and compute R² against data."""
    data = load_dataset()
    n = len(data)
    t_naive = np.linspace(T_MIN, T_MAX, n)

    # Fit theta, M, X to minimise MSE assuming row i ↔ t_naive[i]
    def mse_roworder(params):
        predicted = curve_at_t(t_naive, *params)
        return float(np.mean(np.sum((data - predicted) ** 2, axis=1)))

    result = differential_evolution(
        mse_roworder,
        bounds=BOUNDS,
        seed=42,
        popsize=10,
        maxiter=150,
        tol=1e-6,
        updating="immediate",
        workers=1,
    )
    best_params = result.x
    predicted = curve_at_t(t_naive, *best_params)
    ss_res = np.sum((data - predicted) ** 2)
    ss_tot = np.sum((data - data.mean(axis=0)) ** 2)
    r2 = 1.0 - ss_res / ss_tot

    print("=== Claim 1: Naive row-order regression ===")
    print(f"  Best-fit params (theta, M, X): ({best_params[0]:.4f}, {best_params[1]:.4f}, {best_params[2]:.4f})")
    print(f"  MSE: {result.fun:.6e}")
    print(f"  R² (vs raw data): {r2:.6f}")
    print()
    return r2


# ─── Claim 2: KD-tree alone (no closed-form), loss plateau ──────────────────

def kdtree_only_fit():
    """Run the KD-tree only fit and report convergence loss."""
    data = load_dataset()
    print("=== Claim 2: KD-tree nearest-neighbour fit (no closed-form) ===")
    print(f"  Running differential_evolution with {N_CURVE_SAMPLES:,} sampled curve points...")
    kdt_params, kdt_loss = fit_kdtree(data, n_samples=N_CURVE_SAMPLES)
    print(f"  Fitted theta = {kdt_params[0]:.6f} deg")
    print(f"  Fitted M     = {kdt_params[1]:.6f}")
    print(f"  Fitted X     = {kdt_params[2]:.6f}")
    print(f"  Mean nearest-neighbour Manhattan/L1 loss = {kdt_loss:.6e}")
    print()
    return kdt_loss


if __name__ == "__main__":
    r2 = naive_row_order_fit()
    kdt_loss = kdtree_only_fit()
    print("─" * 60)
    print("SUMMARY (values to compare with README claims):")
    print(f"  Naive row-order regression R²:  {r2:.4f}")
    print(f"  KD-tree alone convergence loss: {kdt_loss:.4e}")
    print()
    if r2 < 0:
        print("  → R² < 0: row-order regression performs WORSE than the mean — 'negative R²' claim is CONFIRMED.")
    elif r2 < 0.5:
        print("  → R² positive but low — soften README claim to qualitative statement.")
    
    if kdt_loss < 2e-3:
        print(f"  → KD-tree loss {kdt_loss:.4e} is consistent with README's '~1.7 × 10⁻³' claim — CONFIRMED.")
    else:
        print(f"  → KD-tree loss differs from README claim — README needs correcting to {kdt_loss:.2e}.")
