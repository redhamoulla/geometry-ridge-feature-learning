#!/usr/bin/env python3
"""Controlled non-scalar collision experiment for the geometric observation register.

Constructs two distinct relative quadratic data (G=I, Gamma, alpha) whose common scalar
summaries are exactly matched: trace contraction, log-det information gain,
Euclidean displacement norm, post-insertion Cook norm, and effect on one target.
Random orthogonal rotations remove coordinate artefacts. The full intrinsic orbit
signature (contraction spectrum and effort energy in its eigenspaces) separates
both mechanisms, while any classifier using the matched scalar summaries is at
chance.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(os.environ.get("EXPERIMENT_OUTPUT_DIR", str(Path(__file__).resolve().parent)))
OUT.mkdir(parents=True, exist_ok=True)

# Two contraction spectra with exactly identical trace and determinant of I-C.
C_A = np.array([0.8, 0.6, 0.2], dtype=float)
_a = 0.15
_sum_rem = float(C_A.sum() - _a)
_prod_rem = float(np.prod(1.0 - C_A) / (1.0 - _a))
# If b+c=s and (1-b)(1-c)=p, then bc=p-1+s.
_bc = _prod_rem - 1.0 + _sum_rem
_disc = _sum_rem * _sum_rem - 4.0 * _bc
_b = (_sum_rem + np.sqrt(_disc)) / 2.0
_c = (_sum_rem - np.sqrt(_disc)) / 2.0
C_B = np.array([_b, _c, _a], dtype=float)

# Displacements with identical Euclidean norm, identical first target effect,
# and identical post-insertion metric norm d^T (I-C)^(-1) d.
_target_first_sq = 0.01
_remaining_sq = 0.99
_target_cook = 1.78

def _remaining_components(c: np.ndarray) -> tuple[float, float]:
    w = 1.0 / (1.0 - c)
    x2sq = (_target_cook - w[0] * _target_first_sq - w[2] * _remaining_sq) / (w[1] - w[2])
    x3sq = _remaining_sq - x2sq
    if x2sq <= 0 or x3sq <= 0:
        raise RuntimeError("Invalid matched displacement construction")
    return float(np.sqrt(x2sq)), float(np.sqrt(x3sq))

_a2, _a3 = _remaining_components(C_A)
_b2, _b3 = _remaining_components(C_B)
D_A = np.array([0.1, _a2, _a3], dtype=float)
D_B = np.array([0.1, _b2, _b3], dtype=float)
TARGET = np.array([1.0, 0.0, 0.0])


def haar_orthogonal(rng: np.random.Generator, p: int) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(p, p)))
    signs = np.sign(np.diag(r)); signs[signs == 0] = 1.0
    q = q @ np.diag(signs)
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1.0
    return q


def build_row(kind: str, pair_id: int, q: np.ndarray) -> dict:
    c = C_A if kind == "A" else C_B
    d0 = D_A if kind == "A" else D_B
    # B = C(I-C)^(-1), G = I.  d = -(I+B)^(-1) alpha.
    bdiag = c / (1.0 - c)
    alpha0 = -(1.0 + bdiag) * d0

    C = q @ np.diag(c) @ q.T
    d = q @ d0
    alpha = q @ alpha0
    target = q @ TARGET

    # Scalar summaries frequently used by scalar attribution diagnostics.
    trace_c = float(np.trace(C))
    info = float(-0.5 * np.linalg.slogdet(np.eye(3) - C)[1])
    norm_d = float(np.linalg.norm(d))
    cook_post = float(d @ np.linalg.solve(np.eye(3) - C, d))
    target_effect = float(target @ d)

    # Intrinsic orbit signature. Since eigenvalues are distinct, the squared
    # projected effort in each eigenspace is a complete coordinate-free feature.
    eigvals, eigvecs = np.linalg.eigh(C)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    effort_energy = (eigvecs.T @ alpha) ** 2

    row = {
        "pair_id": pair_id,
        "kind": kind,
        "label": 0 if kind == "A" else 1,
        "trace_contraction": trace_c,
        "information_gain": info,
        "displacement_norm": norm_d,
        "cook_post_norm_sq": cook_post,
        "target_effect": target_effect,
    }
    for j in range(3):
        row[f"contraction_eigenvalue_{j+1}"] = float(eigvals[j])
        row[f"effort_energy_{j+1}"] = float(effort_energy[j])
    return row


def grouped_auc(df: pd.DataFrame, columns: list[str]) -> float:
    X = df[columns].to_numpy(float)
    y = df["label"].to_numpy(int)
    groups = df["pair_id"].to_numpy(int)
    # Round scalar collisions to remove irrelevant floating-point differences.
    X = np.round(X, 10)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    cv = GroupKFold(n_splits=10)
    model = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced")
    pred = cross_val_predict(model, Xs, y, groups=groups, cv=cv, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, pred))


def main() -> None:
    rng = np.random.default_rng(20260803)
    rows = []
    n_pairs = 400
    for pair_id in range(n_pairs):
        q = haar_orthogonal(rng, 3)
        rows.append(build_row("A", pair_id, q))
        rows.append(build_row("B", pair_id, q))
    df = pd.DataFrame(rows)

    scalar_cols = [
        "trace_contraction",
        "information_gain",
        "displacement_norm",
        "cook_post_norm_sq",
        "target_effect",
    ]
    orbit_cols = [
        "contraction_eigenvalue_1", "contraction_eigenvalue_2", "contraction_eigenvalue_3",
        "effort_energy_1", "effort_energy_2", "effort_energy_3",
    ]
    spectral_cols = orbit_cols[:3]
    effort_cols = orbit_cols[3:]

    aucs = {
        "matched_scalar_summaries": grouped_auc(df, scalar_cols),
        "contraction_spectrum_only": grouped_auc(df, spectral_cols),
        "effort_distribution_only": grouped_auc(df, effort_cols),
        "full_orbit_signature": grouped_auc(df, orbit_cols),
    }

    pair_diff = df.pivot(index="pair_id", columns="kind", values=scalar_cols)
    max_scalar_diff = {}
    for col in scalar_cols:
        max_scalar_diff[col] = float(np.max(np.abs(pair_diff[(col, "A")] - pair_diff[(col, "B")])))

    summary = {
        "n_pairs": n_pairs,
        "class_A_contraction_spectrum": C_A.tolist(),
        "class_B_contraction_spectrum": C_B.tolist(),
        "class_A_displacement": D_A.tolist(),
        "class_B_displacement": D_B.tolist(),
        "max_absolute_pairwise_scalar_difference": max_scalar_diff,
        "auc": aucs,
    }
    df.to_csv(OUT / "nonscalar_orbit_collision_points.csv", index=False)
    with open(OUT / "nonscalar_orbit_collision_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Figure: matched scalar projections versus distinct intrinsic spectra.
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9))
    x = np.arange(5)
    vals_a = df[df.kind == "A"][scalar_cols].iloc[0].to_numpy(float)
    vals_b = df[df.kind == "B"][scalar_cols].iloc[0].to_numpy(float)
    # Normalize only for display; equality is preserved.
    scale = np.maximum(np.abs(vals_a), 1e-12)
    axes[0].plot(x, vals_a / scale, marker="o", label="registre A")
    axes[0].plot(x, vals_b / scale, marker="s", linestyle="--", label="registre B")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(["trace", "info", "$\\|d\\|$", "Cook", "cible"], rotation=20)
    axes[0].set_ylabel("Valeur normalisée")
    axes[0].set_title("Scalarisations exactement appariées")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(alpha=0.2)

    k = np.arange(1, 4)
    axes[1].plot(k, np.sort(C_A)[::-1], marker="o", label="spectre A")
    axes[1].plot(k, np.sort(C_B)[::-1], marker="s", linestyle="--", label="spectre B")
    axes[1].set_xticks(k)
    axes[1].set_xlabel("Direction propre")
    axes[1].set_ylabel("Valeur propre de contraction")
    axes[1].set_title("Le registre non scalaire les distingue")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "nonscalar_orbit_collision.pdf")
    plt.close(fig)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
