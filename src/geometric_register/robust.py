"""Bounded-geometric ridge prototype used in robustness experiments."""
from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def ridge_fit(X: ArrayLike, y: ArrayLike, alpha: float, weights: ArrayLike | None = None) -> FloatArray:
    X = np.asarray(X, float); y = np.asarray(y, float)
    w = np.ones(len(y)) if weights is None else np.asarray(weights, float)
    gram = X.T @ (w[:, None] * X) + alpha * np.eye(X.shape[1])
    return np.linalg.solve(gram, X.T @ (w * y))


def huber_weight(z: ArrayLike, kappa: float) -> FloatArray:
    z = np.asarray(z, float); a = np.abs(z)
    return np.where(a <= kappa, 1.0, kappa / np.maximum(a, 1e-15))


def geometric_weight(X: ArrayLike, quantile: float = 0.95) -> tuple[FloatArray, float]:
    X = np.asarray(X, float); norms = np.linalg.norm(X, axis=1)
    kappa = float(np.quantile(norms, quantile))
    w = np.minimum(1.0, (kappa / np.maximum(norms, 1e-15)) ** 2)
    return w, kappa


def fit_bgr(
    X: ArrayLike, y: ArrayLike, alpha: float,
    geometric_quantile: float = 0.95, residual_kappa: float = 3.0,
    max_iter: int = 30, tol: float = 1e-8,
) -> tuple[FloatArray, dict[str, FloatArray | float]]:
    """Fit ridge with separately bounded contraction and innovation budgets."""
    X = np.asarray(X, float); y = np.asarray(y, float)
    wx, kx = geometric_weight(X, geometric_quantile)
    beta = ridge_fit(X, y, alpha, wx)
    wy = np.ones(len(y)); scale = 1.0
    for _ in range(max_iter):
        residual = y - X @ beta
        med = np.median(residual)
        scale = 1.4826 * np.median(np.abs(residual - med)) + 1e-12
        wy = huber_weight(residual / scale, residual_kappa)
        new = ridge_fit(X, y, alpha, wx * wy)
        if np.linalg.norm(new - beta) <= tol * (1.0 + np.linalg.norm(beta)):
            beta = new; break
        beta = new
    return beta, {"geometric_weights": wx, "residual_weights": wy, "kappa_x": kx, "scale": scale}
