"""Core linear-algebra objects for the conditional geometric register."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def _sym(a: ArrayLike) -> FloatArray:
    x = np.asarray(a, dtype=float)
    return 0.5 * (x + x.T)


def whiten_triplet(G: ArrayLike, Gamma: ArrayLike, alpha: ArrayLike) -> tuple[FloatArray, FloatArray]:
    """Whiten ``(G, Gamma, alpha)`` into the orthogonal orbit pair ``(B,b)``."""
    G = _sym(G); Gamma = _sym(Gamma); alpha = np.asarray(alpha, dtype=float)
    vals, vecs = np.linalg.eigh(G)
    if vals.min() <= 0:
        raise ValueError("G must be positive definite")
    invsqrt = (vecs / np.sqrt(vals)) @ vecs.T
    return _sym(invsqrt @ Gamma @ invsqrt), invsqrt @ alpha


@dataclass(frozen=True)
class SpectralComponent:
    eigenvalue: float
    multiplicity: int
    effort_energy: float


def orbit_signature(B: ArrayLike, b: ArrayLike, tol: float = 1e-9) -> list[SpectralComponent]:
    """Complete orthogonal-orbit signature of a symmetric pair ``(B,b)``."""
    B = _sym(B); b = np.asarray(b, dtype=float)
    vals, vecs = np.linalg.eigh(B)
    groups: list[list[int]] = []
    for idx, value in enumerate(vals):
        if not groups or abs(value - vals[groups[-1][0]]) > tol * max(1.0, abs(value)):
            groups.append([idx])
        else:
            groups[-1].append(idx)
    out = []
    for group in groups:
        basis = vecs[:, group]
        out.append(SpectralComponent(float(vals[group[0]]), len(group), float(np.linalg.norm(basis.T @ b) ** 2)))
    return out


def contraction_operator(B: ArrayLike) -> FloatArray:
    """Relative contraction ``C=B(I+B)^{-1}``."""
    B = _sym(B)
    return np.linalg.solve(np.eye(B.shape[0]) + B, B).T


def information_gain(B: ArrayLike) -> float:
    """Half log-determinant information gain."""
    sign, logdet = np.linalg.slogdet(np.eye(np.asarray(B).shape[0]) + _sym(B))
    if sign <= 0:
        raise ValueError("I+B must be positive definite")
    return 0.5 * float(logdet)


def local_displacement(B: ArrayLike, b: ArrayLike) -> FloatArray:
    """Whitened finite local displacement ``-(I+B)^{-1}b``."""
    B = _sym(B); b = np.asarray(b, dtype=float)
    return -np.linalg.solve(np.eye(B.shape[0]) + B, b)


def register_diagnostics(B: ArrayLike, b: ArrayLike) -> dict[str, float]:
    B = _sym(B); b = np.asarray(b, dtype=float)
    C = contraction_operator(B); d = local_displacement(B, b)
    return {
        "trace_contraction": float(np.trace(C)),
        "information_gain": information_gain(B),
        "effort_norm_sq": float(b @ b),
        "displacement_norm_sq": float(d @ d),
        "post_metric_displacement_sq": float(d @ (np.eye(B.shape[0]) + B) @ d),
    }


def observation_register(X_context: ArrayLike, x: ArrayLike, residual: float, ridge: float) -> tuple[FloatArray, FloatArray]:
    """Rank-one register for scalar linear regression."""
    X = np.asarray(X_context, dtype=float); x = np.asarray(x, dtype=float)
    G = X.T @ X + ridge * np.eye(X.shape[1])
    Gamma = np.outer(x, x); alpha = -residual * x
    return whiten_triplet(G, Gamma, alpha)
