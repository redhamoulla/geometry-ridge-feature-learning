#!/usr/bin/env python3
"""Controlled nonlinear experiment for the dynamic per-observation geometric register.

Trains a small ReLU MLP on clean Friedman-1 anchor data and evaluates held-out
probe observations under five regimes: ordinary clean, rare clean, vertical-label
corruption, leverage/covariate corruption, and mixed corruption. At several
training epochs it computes the conditional contraction and natural effort of
each probe relative to the collective Gauss--Newton geometry of the anchor set.

The script is deterministic conditional on the replicate seed and writes CSV/JSON
summaries plus two PDF figures used by the paper.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.datasets import make_friedman1
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class Config:
    n_anchor: int = 500
    n_probe_pool: int = 600
    n_test: int = 1000
    n_features: int = 10
    hidden: int = 8
    noise: float = 1.0
    epochs: int = 300
    checkpoints: Tuple[int, ...] = (0, 25, 100, 300)
    lr: float = 0.02
    weight_decay: float = 1e-4
    n_per_group: int = 60
    label_amplitude: float = 4.0
    leverage_amplitude: float = 10.0
    ridge_metric: float = 1e-2
    replicates: int = 10


class SmallMLP(torch.nn.Module):
    def __init__(self, d: int, h: int):
        super().__init__()
        self.fc1 = torch.nn.Linear(d, h)
        self.fc2 = torch.nn.Linear(h, 1)
        torch.nn.init.normal_(self.fc1.weight, mean=0.0, std=0.18)
        torch.nn.init.zeros_(self.fc1.bias)
        torch.nn.init.normal_(self.fc2.weight, mean=0.0, std=0.18)
        torch.nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x))).squeeze(-1)


def flatten_params(model: SmallMLP) -> np.ndarray:
    with torch.no_grad():
        W = model.fc1.weight.detach().cpu().numpy().copy()
        b = model.fc1.bias.detach().cpu().numpy().copy()
        a = model.fc2.weight.detach().cpu().numpy().reshape(-1).copy()
        c = np.array([float(model.fc2.bias.detach().cpu().numpy()[0])])
    return np.concatenate([W.ravel(), b, a, c])


def unpack(theta: np.ndarray, d: int, h: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    k = 0
    W = theta[k:k+h*d].reshape(h, d); k += h*d
    b = theta[k:k+h]; k += h
    a = theta[k:k+h]; k += h
    c = float(theta[k])
    return W, b, a, c


def forward_and_jacobian(theta: np.ndarray, X: np.ndarray, d: int, h: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return scalar predictions and per-sample Jacobian wrt flattened parameters."""
    W, b, a, c = unpack(theta, d, h)
    Z = X @ W.T + b
    mask = (Z > 0.0).astype(np.float64)
    H = np.maximum(Z, 0.0)
    pred = H @ a + c

    # d f / d W_{k,j} = a_k 1[z_k>0] x_j
    coeff = mask * a[None, :]
    JW = np.einsum("nh,nd->nhd", coeff, X).reshape(X.shape[0], h*d)
    Jb = coeff
    Ja = H
    Jc = np.ones((X.shape[0], 1), dtype=np.float64)
    J = np.concatenate([JW, Jb, Ja, Jc], axis=1)
    return pred, J


def robust_scale(resid: np.ndarray) -> float:
    med = np.median(resid)
    mad = 1.4826 * np.median(np.abs(resid - med))
    return float(max(mad, 0.08))


def make_groups(
    X_pool: np.ndarray,
    y_pool: np.ndarray,
    cfg: Config,
    rng: np.random.Generator,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    n = X_pool.shape[0]
    norms = np.linalg.norm(X_pool, axis=1)
    rare_candidates = np.argsort(norms)[-max(2 * cfg.n_per_group, cfg.n_per_group):]
    rare_idx = rng.choice(rare_candidates, size=cfg.n_per_group, replace=False)

    remaining = np.setdiff1d(np.arange(n), rare_idx, assume_unique=False)
    rng.shuffle(remaining)
    chunks = np.array_split(remaining[: 4 * cfg.n_per_group], 4)
    ordinary_idx, label_idx, leverage_idx, mixed_idx = chunks

    groups: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    groups["ordinaire propre"] = (X_pool[ordinary_idx].copy(), y_pool[ordinary_idx].copy())
    groups["rare propre"] = (X_pool[rare_idx].copy(), y_pool[rare_idx].copy())

    Xlab = X_pool[label_idx].copy(); ylab = y_pool[label_idx].copy()
    ylab += cfg.label_amplitude * rng.choice([-1.0, 1.0], size=ylab.size)
    groups["label aberrant"] = (Xlab, ylab)

    Xlev = X_pool[leverage_idx].copy(); ylev = y_pool[leverage_idx].copy()
    directions = rng.normal(size=Xlev.shape)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True) + 1e-12
    Xlev += cfg.leverage_amplitude * directions
    groups["covariable aberrante"] = (Xlev, ylev)

    Xmix = X_pool[mixed_idx].copy(); ymix = y_pool[mixed_idx].copy()
    directions = rng.normal(size=Xmix.shape)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True) + 1e-12
    Xmix += cfg.leverage_amplitude * directions
    ymix += cfg.label_amplitude * rng.choice([-1.0, 1.0], size=ymix.size)
    groups["mixte"] = (Xmix, ymix)
    return groups


def compute_register(
    theta: np.ndarray,
    X_anchor: np.ndarray,
    y_anchor: np.ndarray,
    groups: Dict[str, Tuple[np.ndarray, np.ndarray]],
    cfg: Config,
) -> Tuple[pd.DataFrame, Dict[str, float], np.ndarray]:
    pred_a, J_a = forward_and_jacobian(theta, X_anchor, cfg.n_features, cfg.hidden)
    resid_a = pred_a - y_anchor
    sigma = robust_scale(resid_a)
    p = J_a.shape[1]
    G = cfg.ridge_metric * np.eye(p) + (J_a.T @ J_a) / (sigma * sigma)
    # Cholesky solve is stable and avoids materializing an inverse.
    L = np.linalg.cholesky(G)

    rows: List[dict] = []
    all_mu_clean = []
    all_q_clean = []
    jacobians = []
    for group, (Xg, yg) in groups.items():
        pred, J = forward_and_jacobian(theta, Xg, cfg.n_features, cfg.hidden)
        r = pred - yg
        # Solve G^{-1} J^T in one block.
        tmp = np.linalg.solve(L, J.T)
        sol = np.linalg.solve(L.T, tmp)
        mu = np.einsum("np,pn->n", J, sol) / (sigma * sigma)
        mu = np.maximum(mu, 0.0)
        contraction = mu / (1.0 + mu)
        z = r / sigma
        natural_effort = np.abs(z) * np.sqrt(mu)
        information = 0.5 * np.log1p(mu)
        influence_local = natural_effort / (1.0 + mu)
        jacobians.append(J)
        if group in ("ordinaire propre", "rare propre"):
            all_mu_clean.extend(mu.tolist())
            all_q_clean.extend(natural_effort.tolist())
        for k in range(len(r)):
            rows.append({
                "group": group,
                "mu": float(mu[k]),
                "contraction": float(contraction[k]),
                "information": float(information[k]),
                "standardized_residual": float(abs(z[k])),
                "natural_effort": float(natural_effort[k]),
                "local_displacement_norm": float(influence_local[k]),
            })
    df = pd.DataFrame(rows)
    thresholds = {
        "sigma": sigma,
        "kappa_x": float(np.quantile(np.asarray(all_mu_clean), 0.95)),
        "kappa_y": float(np.quantile(np.asarray(all_q_clean), 0.95)),
    }
    J_concat = np.concatenate(jacobians, axis=0)
    return df, thresholds, J_concat


def train_one(seed: int, cfg: Config) -> Tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    X, y = make_friedman1(
        n_samples=cfg.n_anchor + cfg.n_probe_pool + cfg.n_test,
        n_features=cfg.n_features,
        noise=cfg.noise,
        random_state=seed,
    )
    X_anchor = X[: cfg.n_anchor]
    y_anchor = y[: cfg.n_anchor]
    X_pool = X[cfg.n_anchor : cfg.n_anchor + cfg.n_probe_pool]
    y_pool = y[cfg.n_anchor : cfg.n_anchor + cfg.n_probe_pool]
    X_test = X[-cfg.n_test :]
    y_test = y[-cfg.n_test :]

    x_scaler = StandardScaler().fit(X_anchor)
    X_anchor = x_scaler.transform(X_anchor)
    X_pool = x_scaler.transform(X_pool)
    X_test = x_scaler.transform(X_test)
    y_mean = float(np.mean(y_anchor)); y_std = float(np.std(y_anchor))
    y_anchor = (y_anchor - y_mean) / y_std
    y_pool = (y_pool - y_mean) / y_std
    y_test = (y_test - y_mean) / y_std

    groups = make_groups(X_pool, y_pool, cfg, rng)

    torch.manual_seed(seed)
    model = SmallMLP(cfg.n_features, cfg.hidden).double()
    Xa_t = torch.from_numpy(X_anchor).double()
    ya_t = torch.from_numpy(y_anchor).double()
    Xt_t = torch.from_numpy(X_test).double()
    yt_t = torch.from_numpy(y_test).double()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    records: List[pd.DataFrame] = []
    checkpoint_j: Dict[int, np.ndarray] = {}
    checkpoint_c: Dict[int, np.ndarray] = {}
    test_mse: Dict[int, float] = {}

    for epoch in range(cfg.epochs + 1):
        if epoch in cfg.checkpoints:
            theta = flatten_params(model)
            reg, thresholds, J_probe = compute_register(theta, X_anchor, y_anchor, groups, cfg)
            reg["epoch"] = epoch
            reg["seed"] = seed
            kx = max(thresholds["kappa_x"], 1e-12)
            ky = max(thresholds["kappa_y"], 1e-12)
            mu_clip = np.minimum(reg["mu"].to_numpy(), kx)
            reg["contraction_clipped"] = mu_clip / (1.0 + mu_clip)
            reg["effort_clipped"] = np.minimum(reg["natural_effort"].to_numpy(), ky)
            records.append(reg)
            checkpoint_j[epoch] = J_probe
            checkpoint_c[epoch] = reg["contraction"].to_numpy()
            with torch.no_grad():
                test_mse[epoch] = float(torch.mean((model(Xt_t) - yt_t) ** 2).item())
        if epoch == cfg.epochs:
            break
        opt.zero_grad(set_to_none=True)
        pred = model(Xa_t)
        loss = 0.5 * torch.mean((pred - ya_t) ** 2)
        loss.backward()
        opt.step()

    df = pd.concat(records, ignore_index=True)
    e0 = cfg.checkpoints[0]; ef = cfg.checkpoints[-1]
    J0 = checkpoint_j[e0]; Jf = checkpoint_j[ef]
    per_point_drift = np.linalg.norm(Jf - J0, axis=1) / (np.linalg.norm(J0, axis=1) + 1e-10)
    rho = float(spearmanr(checkpoint_c[e0], checkpoint_c[ef]).statistic)
    meta = {
        "seed": seed,
        "test_mse_initial": test_mse[e0],
        "test_mse_final": test_mse[ef],
        "median_relative_jacobian_drift": float(np.median(per_point_drift)),
        "q90_relative_jacobian_drift": float(np.quantile(per_point_drift, 0.90)),
        "spearman_contraction_initial_final": rho,
    }
    return df, meta


def aggregate_and_plot(all_df: pd.DataFrame, meta_df: pd.DataFrame, out_dir: Path, cfg: Config) -> dict:
    group_order = ["ordinaire propre", "rare propre", "label aberrant", "covariable aberrante", "mixte"]
    final = all_df[all_df["epoch"] == cfg.checkpoints[-1]].copy()
    summary = (
        final.groupby("group")
        .agg(
            contraction_median=("contraction", "median"),
            contraction_q90=("contraction", lambda s: float(np.quantile(s, 0.90))),
            effort_median=("natural_effort", "median"),
            effort_q90=("natural_effort", lambda s: float(np.quantile(s, 0.90))),
            information_median=("information", "median"),
            displacement_median=("local_displacement_norm", "median"),
            contraction_clipped_max=("contraction_clipped", "max"),
            effort_clipped_max=("effort_clipped", "max"),
        )
        .reindex(group_order)
    )
    summary.to_csv(out_dir / "nonlinear_register_group_summary.csv")

    dynamics = (
        all_df.groupby(["epoch", "group"])
        .agg(
            contraction_median=("contraction", "median"),
            effort_median=("natural_effort", "median"),
        )
        .reset_index()
    )
    dynamics.to_csv(out_dir / "nonlinear_register_dynamics.csv", index=False)
    all_df.to_csv(out_dir / "nonlinear_register_point_scores.csv", index=False)
    meta_df.to_csv(out_dir / "nonlinear_register_replicates.csv", index=False)

    # Figure 1: final passport.
    fig, ax = plt.subplots(figsize=(7.2, 4.7))
    markers = ["o", "s", "^", "D", "P"]
    for group, marker in zip(group_order, markers):
        sub = final[final["group"] == group]
        # Plot a deterministic subsample to keep the vector PDF light.
        sub = sub.iloc[::max(1, len(sub)//180)]
        ax.scatter(sub["contraction"], sub["natural_effort"], s=16, alpha=0.48, marker=marker, label=group)
    ax.set_xlabel("Contraction conditionnelle $c_i$")
    ax.set_ylabel("Effort naturel $\\|G^{-1/2}\\alpha_i\\|$")
    ax.set_title("Registre non linéaire au terme de l'apprentissage")
    ax.legend(fontsize=8, frameon=False, ncol=2)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "nonlinear_register_passport.pdf")
    plt.close(fig)

    # Figure 2: dynamics of contraction and effort.
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    for group in group_order:
        sub = dynamics[dynamics["group"] == group]
        axes[0].plot(sub["epoch"], sub["contraction_median"], marker="o", linewidth=1.4, label=group)
        axes[1].plot(sub["epoch"], sub["effort_median"], marker="o", linewidth=1.4, label=group)
    axes[0].set_xlabel("Époque"); axes[0].set_ylabel("Contraction médiane")
    axes[1].set_xlabel("Époque"); axes[1].set_ylabel("Effort naturel médian")
    for ax in axes: ax.grid(alpha=0.2)
    axes[0].set_title("Géométrie mobile")
    axes[1].set_title("Innovation mobile")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.tight_layout(rect=[0, 0.13, 1, 1])
    fig.savefig(out_dir / "nonlinear_register_dynamics.pdf")
    plt.close(fig)

    result = {
        "config": cfg.__dict__,
        "groups_final": summary.reset_index().to_dict(orient="records"),
        "replicate_dynamics": {
            "test_mse_final_mean": float(meta_df["test_mse_final"].mean()),
            "test_mse_final_std": float(meta_df["test_mse_final"].std(ddof=1)),
            "median_relative_jacobian_drift_mean": float(meta_df["median_relative_jacobian_drift"].mean()),
            "median_relative_jacobian_drift_std": float(meta_df["median_relative_jacobian_drift"].std(ddof=1)),
            "q90_relative_jacobian_drift_mean": float(meta_df["q90_relative_jacobian_drift"].mean()),
            "spearman_contraction_initial_final_mean": float(meta_df["spearman_contraction_initial_final"].mean()),
            "spearman_contraction_initial_final_std": float(meta_df["spearman_contraction_initial_final"].std(ddof=1)),
        },
    }
    with open(out_dir / "nonlinear_register_summary.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return result


def main() -> None:
    cfg = Config()
    out_dir = Path(os.environ.get("EXPERIMENT_OUTPUT_DIR", str(Path(__file__).resolve().parent)))
    out_dir.mkdir(parents=True, exist_ok=True)
    frames: List[pd.DataFrame] = []
    metas: List[dict] = []
    for r in range(cfg.replicates):
        df, meta = train_one(9100 + r, cfg)
        frames.append(df); metas.append(meta)
        print(f"replicate {r+1}/{cfg.replicates}: mse={meta['test_mse_final']:.4f}, drift={meta['median_relative_jacobian_drift']:.3f}")
    all_df = pd.concat(frames, ignore_index=True)
    meta_df = pd.DataFrame(metas)
    result = aggregate_and_plot(all_df, meta_df, out_dir, cfg)
    print(json.dumps(result["replicate_dynamics"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
