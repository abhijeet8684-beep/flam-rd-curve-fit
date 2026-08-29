"""Deterministic regression and generalisation checks for the fitting pipeline."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fit_curve import DATA_PATH, T_MAX, T_MIN, curve_at_t, fit_primary_de_rotation, load_dataset


def test_curve_formula_matches_independent_math() -> None:
    """Check one curve point against a direct stdlib implementation of the equation."""
    t, theta_deg, m, x_offset = 17.25, 32.5, -0.021, 41.0
    theta = math.radians(theta_deg)
    wave = math.exp(m * abs(t)) * math.sin(0.3 * t)
    expected = np.array([
        t * math.cos(theta) - wave * math.sin(theta) + x_offset,
        42.0 + t * math.sin(theta) + wave * math.cos(theta),
    ])
    actual = curve_at_t(t, theta_deg, m, x_offset)[0]
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-10)


def test_real_csv_has_expected_shape_and_finite_plausible_values() -> None:
    """Guard against a missing, corrupted, or incorrectly parsed assignment dataset."""
    data = load_dataset(DATA_PATH)
    assert data.shape == (1_500, 2)
    assert np.isfinite(data).all()
    assert data[:, 0].min() > -100 and data[:, 0].max() < 200
    assert data[:, 1].min() > -100 and data[:, 1].max() < 200


def write_synthetic_csv(path: Path, parameters: tuple[float, float, float], seed: int) -> None:
    """Create an unordered synthetic point cloud using a known parameter set."""
    rng = np.random.default_rng(seed)
    t = rng.uniform(T_MIN, T_MAX, size=240)
    points = curve_at_t(t, *parameters)
    rng.shuffle(points)
    np.savetxt(path, points, delimiter=",", header="x,y", comments="")


@pytest.mark.parametrize(
    ("parameters", "seed"),
    [((12.0, -0.03, 20.0), 17), ((45.0, 0.04, 90.0), 29)],
)
def test_closed_form_pipeline_recovers_unseen_synthetic_parameters(
    tmp_path: Path, parameters: tuple[float, float, float], seed: int
) -> None:
    """Recover two non-assignment parameter sets through the actual DE-plus-polish pipeline."""
    csv_path = tmp_path / "synthetic.csv"
    write_synthetic_csv(csv_path, parameters, seed)
    recovered, loss = fit_primary_de_rotation(load_dataset(csv_path), seed=seed, maxiter=160)
    assert loss < 1e-8
    assert abs(recovered[0] - parameters[0]) <= 0.1
    assert abs(recovered[1] - parameters[1]) <= 1e-3
    assert abs(recovered[2] - parameters[2]) <= 0.5


@pytest.mark.slow
def test_real_data_pipeline_recovers_reported_parameters() -> None:
    """Catch a regression that changes the established assignment answer."""
    recovered, loss = fit_primary_de_rotation(load_dataset(DATA_PATH), seed=2026)
    assert loss < 1e-4
    assert abs(recovered[0] - 30.0) <= 0.1
    assert abs(recovered[1] - 0.03) <= 1e-3
    assert abs(recovered[2] - 55.0) <= 0.5
