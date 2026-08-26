# Parametric Curve Fitting

## Problem

This project fits the 1,500 unordered `(x, y)` points in `data/xy_data.csv` to this parametric curve:

```text
x = t*cos(theta) - exp(M*abs(t))*sin(0.3*t)*sin(theta) + X
y = 42 + t*sin(theta) + exp(M*abs(t))*sin(0.3*t)*cos(theta)
```

The valid ranges are `6 < t < 60`, `0 < theta < 50 degrees`, `-0.05 < M < 0.05`, and `0 < X < 100`. The program receives `theta` in degrees and converts it to radians before evaluating the curve.

The original assignment brief is included at `data/assignment_brief.pdf` for reference.

## Why This Works: Geometric Rotation Decoupling

The parametric equations can be written in compact matrix form as a 2D rotation followed by a translation:

$$\begin{pmatrix} x - X \\ y - 42 \end{pmatrix} = \begin{pmatrix} \cos(\theta) & -\sin(\theta) \\ \sin(\theta) & \cos(\theta) \end{pmatrix} \begin{pmatrix} t \\ e^{M|t|}\sin(0.3t) \end{pmatrix}$$

By applying the inverse rotation $R(-\theta)$ to the centered coordinates $(x_i - X, y_i - 42)$, we algebraically decouple the longitudinal parameter $t$ from the transverse oscillation:

$$u_i = (x_i - X)\cos(\theta) + (y_i - 42)\sin(\theta) = t_i$$
$$v_i = -(x_i - X)\sin(\theta) + (y_i - 42)\cos(\theta) = e^{M|t_i|}\sin(0.3t_i)$$

Because $u_i$ recovers each point's parameter $t_i$ in exact closed form, there is no need to approximate point correspondence with discrete nearest-neighbor lookups. The mismatch between actual $v_i$ and predicted $\hat{v}_i = e^{M|u_i|}\sin(0.3u_i)$ gives the exact transverse residual for any trial $(\theta, M, X)$.

## Method

Because the CSV rows are unordered and lack per-point $t$ timestamps, direct pointwise regression against index is impossible.

Using the rotation-decoupling trick above, the optimization pipeline operates as follows:
1. **Global Search via Differential Evolution**: SciPy's `differential_evolution` minimizes the mean absolute transverse residual $\frac{1}{N}\sum |v_i - \hat{v}_i|$ over the parameter space bounds.
2. **Local Least-Squares Polish**: `scipy.optimize.least_squares` refines the solution to sub-micro precision on the exact nonlinear residual vector.
3. **Continuous $t$-Inversion Validation**: For authoritative verification, bounded scalar minimization (`minimize_scalar`) computes the continuous perpendicular Euclidean and Manhattan (L1) reconstruction distances from every supplied point to the true continuous curve.

## Result

The final reported parameters are:

```ini
theta = 30 degrees
M = 0.03
X = 55
```

### Parameter Convergence Evidence

- **Fitted parameters from global search & polish**:
  - $\theta = 29.9999729322^\circ$
  - $M = 0.0299999969$
  - $X = 54.9999982128$
  - Closed-form mean L1 residual: $2.5598054385 \times 10^{-6}$

- **Final reported parameters (rounded / exact)**:
  - $\theta = 30^\circ$, $M = 0.03$, $X = 55$

## Validation & Fit-Quality Metrics

Validation is performed directly on the continuous parametric curve across all 1,500 data points:

| Metric | Mean | 95th Percentile | Maximum |
| :--- | :--- | :--- | :--- |
| **Euclidean Reconstruction Distance** | $1.25615 \times 10^{-5}$ | $2.33709 \times 10^{-5}$ | $3.21073 \times 10^{-5}$ |
| **Manhattan / L1 Reconstruction Distance** | $1.48084 \times 10^{-5}$ | $2.74482 \times 10^{-5}$ | $4.11511 \times 10^{-5}$ |
| **Closed-Form Transverse Residual ($\|v - \hat{v}\|$)** | $1.50483 \times 10^{-5}$ | $2.95795 \times 10^{-5}$ | $4.05114 \times 10^{-5}$ |

> [!NOTE]
> The mean L1 reconstruction distance is $\sim 1.48 \times 10^{-5}$, an improvement over discrete finite-sampling approximations ($\sim 1.70 \times 10^{-3}$), reflecting exact closed-form alignment.

## Final Answer (LaTeX / Desmos)

This is the copyable Desmos parametric expression for $6 \le t \le 60$, submitted per the assignment's required Submission Format of writing or copying the equation in LaTeX format in the README:

```latex
\left(t*\cos(0.5236)-e^{0.03\left|t\right|}\cdot\sin(0.3t)\sin(0.5236)+55,42+t*\sin(0.5236)+e^{0.03\left|t\right|}\cdot\sin(0.3t)\cos(0.5236)\right)
```

## Curve Comparison

![Fitted curve compared with supplied data](results/curve_fit.png)

The blue points are the supplied data and the orange curve uses the final fitted parameters.

### Desmos Curve

![Desmos parametric curve](screenshots/desmos_curve.png)

[Open the interactive Desmos graph](https://www.desmos.com/calculator/evnyyb7znw).

> [!NOTE]
> The saved interactive Desmos graph has been verified to ensure the domain restriction $\{6 \le t \le 60\}$ is retained with no syntax warnings. If manually pasting LaTeX expressions into Desmos, ensure that parametric domain bounds are properly preserved, as external clipboard pasting can occasionally strip curly braces.

## Run in VS Code

Open this folder in VS Code, then run these commands in the integrated PowerShell terminal:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python src\fit_curve.py
```

The script prints full convergence statistics and saves the plot to `results/curve_fit.png`.
