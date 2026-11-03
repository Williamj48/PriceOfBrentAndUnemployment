"""
Project 1: Least Squares -- demo script
Dataset: FRED -- Brent crude oil price (POILBREUSDM) and US unemployment
rate (UNRATE), joined on observation_date (see combined.csv).
Experimental question: Is there a relationship between the price of oil
and the US unemployment rate?

This script solves the SAME linear least-squares problem three ways
(normal equations + Cholesky, QR + back-substitution, built-in solver),
compares them, reports the condition number of A, and then explores a
couple of nonlinear fits (quadratic, cubic) as alternatives to the
simple linear model.

Run with: ../venv/bin/python oil_unemployment_demo.py
"""

import numpy as np
import pandas as pd
from scipy.linalg import cholesky          # Cholesky decomposition (like MATLAB's chol)
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Config: change these two lines to try a different x/y pair from the CSV
# ----------------------------------------------------------------------
DATA_PATH = "data/combined.csv"
X_COL = "POILBREUSDM"   # Brent crude oil price, $/barrel
Y_COL = "UNRATE"        # US unemployment rate, %
# Exclude the COVID shock months: unemployment spiked to ~15% in a single
# month and oil prices briefly went negative, an outlier regime unrelated
# to the normal oil-price/unemployment relationship we're modeling.
EXCLUDE_START = "2020-01-01"
EXCLUDE_END = "2021-01-01"

# ----------------------------------------------------------------------
# 1. Load and prepare data
# ----------------------------------------------------------------------
df = pd.read_csv(DATA_PATH, parse_dates=["observation_date"])
df = df.dropna(subset=[X_COL, Y_COL])   # FRED leaves recent not-yet-released months blank
df = df[(df["observation_date"] < EXCLUDE_START) | (df["observation_date"] > EXCLUDE_END)]

x = df[X_COL].to_numpy(dtype=float)
y = df[Y_COL].to_numpy(dtype=float)

n = len(x)
print(f"Loaded {n} rows. x = {X_COL}, y = {Y_COL}")

# ----------------------------------------------------------------------
# 2. Manually build the design matrix A for a straight-line fit
#    model: y_i ~ b0 + b1 * x_i   =>   A = [1, x_i],  unknowns = [b0, b1]
# ----------------------------------------------------------------------
A = np.column_stack([np.ones(n), x])
b = y

# ----------------------------------------------------------------------
# 3a. Method 1: Normal equations + Cholesky decomposition
#     Solve (A^T A) beta = A^T b
#     A^T A is symmetric positive definite (assuming A has full column
#     rank), so it can be factored as A^T A = R^T R (Cholesky).
# ----------------------------------------------------------------------
AtA = A.T @ A
Atb = A.T @ b

R_chol = cholesky(AtA, lower=False)   # AtA = R_chol.T @ R_chol, R_chol upper triangular


def forward_substitution(L, rhs):
    """Solve L y = rhs for y, where L is lower triangular."""
    m = len(rhs)
    y_out = np.zeros(m)
    for i in range(m):
        y_out[i] = (rhs[i] - L[i, :i] @ y_out[:i]) / L[i, i]
    return y_out


def back_substitution(U, rhs):
    """Solve U x = rhs for x, where U is upper triangular."""
    m = len(rhs)
    x_out = np.zeros(m)
    for i in range(m - 1, -1, -1):
        x_out[i] = (rhs[i] - U[i, i + 1:] @ x_out[i + 1:]) / U[i, i]
    return x_out


# Solve R_chol^T z = Atb (forward sub), then R_chol beta = z (back sub)
z = forward_substitution(R_chol.T, Atb)
beta_chol = back_substitution(R_chol, z)

# ----------------------------------------------------------------------
# 3b. Method 2: QR decomposition + manual back-substitution
#     A = Q R  =>  minimizing ||Ax - b|| reduces to solving
#     R x = Q^T b (R upper triangular, so back-substitution applies)
# ----------------------------------------------------------------------
Q, R_qr = np.linalg.qr(A)          # "economy" QR: Q is n x 2, R_qr is 2 x 2
Qtb = Q.T @ b
beta_qr = back_substitution(R_qr, Qtb)

# ----------------------------------------------------------------------
# 3c. Method 3: Built-in least-squares solver (NumPy's lstsq), the
#     analogue of MATLAB's lsqr / backslash for comparison.
# ----------------------------------------------------------------------
beta_builtin, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)

print("\n--- Coefficients [intercept, slope] ---")
print(f"Normal eqs + Cholesky : {beta_chol}")
print(f"QR + back-substitution: {beta_qr}")
print(f"Built-in lstsq        : {beta_builtin}")
print(f"Max pairwise difference: {np.max(np.abs(beta_chol - beta_builtin)):.3e}")

# ----------------------------------------------------------------------
# 4. Condition number of A (and of A^T A, which is squared -- a nice
#    illustration of why normal equations can be less numerically
#    stable than QR for ill-conditioned problems).
# ----------------------------------------------------------------------
cond_A = np.linalg.cond(A)
cond_AtA = np.linalg.cond(AtA)
print(f"\nCondition number of A      : {cond_A:.3f}")
print(f"Condition number of A^T A  : {cond_AtA:.3f}  (roughly cond(A)^2)")

# ----------------------------------------------------------------------
# 5. Other curve fits: quadratic and cubic (still linear least squares,
#    just with extra columns in A) as alternatives to the simple line.
# ----------------------------------------------------------------------

# Quadratic fit: y ~ b0 + b1*x + b2*x^2
A_quad = np.column_stack([np.ones(n), x, x**2])
beta_quad, *_ = np.linalg.lstsq(A_quad, b, rcond=None)

# Cubic fit: y ~ b0 + b1*x + b2*x^2 + b3*x^3
A_cubic = np.column_stack([np.ones(n), x, x**2, x**3])
beta_cubic, *_ = np.linalg.lstsq(A_cubic, b, rcond=None)

print(f"\nQuadratic fit coefficients [b0, b1, b2]: {beta_quad}")
print(f"Cubic fit coefficients [b0, b1, b2, b3]: {beta_cubic}")

# ----------------------------------------------------------------------
# 6. Plot everything for a sanity check / figure for the report
# ----------------------------------------------------------------------
x_line = np.linspace(x.min(), x.max(), 200)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(x, y, s=12, alpha=0.5, label="data")
axes[0].plot(x_line, beta_builtin[0] + beta_builtin[1] * x_line,
             color="crimson", label="linear least-squares fit")
axes[0].set_xlabel(f"{X_COL} ($/barrel)")
axes[0].set_ylabel(f"{Y_COL} (%)")
axes[0].set_title("Linear fit (normal eqs / QR / lstsq agree)")
axes[0].legend()

axes[1].scatter(x, y, s=12, alpha=0.5, label="data")
axes[1].plot(x_line, beta_quad[0] + beta_quad[1] * x_line + beta_quad[2] * x_line**2,
             color="darkorange", label="quadratic fit")
axes[1].plot(x_line,
             beta_cubic[0] + beta_cubic[1] * x_line + beta_cubic[2] * x_line**2 + beta_cubic[3] * x_line**3,
             color="seagreen", label="cubic fit")
axes[1].set_xlabel(f"{X_COL} ($/barrel)")
axes[1].set_ylabel(f"{Y_COL} (%)")
axes[1].set_title("Nonlinear alternatives")
axes[1].legend()

fig.tight_layout()
fig.savefig("oil_unemployment_demo.png", dpi=200)
print("\nSaved figure to oil_unemployment_demo.png")
