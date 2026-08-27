"""Interactive Parametric Curve Explorer.

Opens a matplotlib window with three sliders (theta, M, X) overlaid on the
actual data points from data/xy_data.csv.  Sliders start at the fitted
answer (theta=30, M=0.03, X=55) so the window opens showing a correct fit.

Run standalone:
    python src/curve_explorer.py

Controls:
    theta slider  -- rotation angle in degrees (0 to 50)
    M     slider  -- exponential envelope coefficient (-0.05 to 0.05)
    X     slider  -- horizontal translation (0 to 100)
    Reset button  -- snap all sliders back to the fitted values

No additional dependencies beyond numpy + matplotlib (already in requirements.txt).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "xy_data.csv"

# ── Fitted (correct) parameter values ─────────────────────────────────────────
INIT_THETA = 30.0   # degrees
INIT_M     = 0.03
INIT_X     = 55.0

# ── Curve parameter bounds ─────────────────────────────────────────────────────
T_MIN, T_MAX = 6.0, 60.0
N_CURVE_PTS  = 2_000  # enough for smooth rendering


def load_data() -> np.ndarray:
    """Load the 1,500 (x,y) data points from CSV."""
    data = np.loadtxt(DATA_PATH, delimiter=",", skiprows=1)
    return data


def make_curve(theta_deg: float, m: float, x_offset: float) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the parametric curve at N_CURVE_PTS values of t."""
    t = np.linspace(T_MIN, T_MAX, N_CURVE_PTS)
    theta = np.deg2rad(theta_deg)
    osc = np.exp(m * np.abs(t)) * np.sin(0.3 * t)
    cx = t * np.cos(theta) - osc * np.sin(theta) + x_offset
    cy = 42.0 + t * np.sin(theta) + osc * np.cos(theta)
    return cx, cy


def main() -> None:
    data = load_data()

    # ── Figure layout ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.subplots_adjust(left=0.1, bottom=0.30)
    fig.canvas.manager.set_window_title("Parametric Curve Explorer — flam-rd-curve-fit")

    # Scatter plot (data)
    scat = ax.scatter(data[:, 0], data[:, 1], s=10, alpha=0.5,
                      color="steelblue", label="Data (1,500 pts)", zorder=2)

    # Initial curve
    cx, cy = make_curve(INIT_THETA, INIT_M, INIT_X)
    (curve_line,) = ax.plot(cx, cy, color="tomato", linewidth=2.0,
                            label=f"θ={INIT_THETA}°  M={INIT_M}  X={INIT_X}", zorder=3)

    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    ax.set_title("Parametric Curve Explorer — drag sliders to explore", fontsize=13)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    legend = ax.legend(loc="upper left", fontsize=10, framealpha=0.9)

    # ── Slider axes ────────────────────────────────────────────────────────────
    ax_theta = fig.add_axes([0.15, 0.20, 0.70, 0.03])
    ax_m     = fig.add_axes([0.15, 0.14, 0.70, 0.03])
    ax_x     = fig.add_axes([0.15, 0.08, 0.70, 0.03])

    sl_theta = Slider(ax_theta, "θ (deg)", 0.0,  50.0,  valinit=INIT_THETA, valstep=0.5)
    sl_m     = Slider(ax_m,     "M",      -0.05,  0.05,  valinit=INIT_M,     valstep=0.001)
    sl_x     = Slider(ax_x,     "X",       0.0, 100.0,  valinit=INIT_X,     valstep=0.5)

    # ── Reset button ───────────────────────────────────────────────────────────
    ax_reset = fig.add_axes([0.80, 0.01, 0.10, 0.04])
    btn_reset = Button(ax_reset, "Reset", color="lightyellow", hovercolor="khaki")

    # ── Update callback ────────────────────────────────────────────────────────
    def update(_):
        theta = sl_theta.val
        m     = sl_m.val
        x     = sl_x.val
        cx, cy = make_curve(theta, m, x)
        curve_line.set_xdata(cx)
        curve_line.set_ydata(cy)
        curve_line.set_label(f"θ={theta:.1f}°  M={m:.3f}  X={x:.1f}")
        legend.remove()
        ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
        fig.canvas.draw_idle()

    def reset(_):
        sl_theta.reset()
        sl_m.reset()
        sl_x.reset()

    sl_theta.on_changed(update)
    sl_m.on_changed(update)
    sl_x.on_changed(update)
    btn_reset.on_clicked(reset)

    plt.show()


if __name__ == "__main__":
    main()
