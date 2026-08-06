#!/usr/bin/env python3
"""Lock-down experiments for the geometric observation register.

1) Exact finite-sample factorization in linear ridge regression:
   - information Shapley equals the context-average log-det contraction;
   - predictive Shapley equals the context-average finite utility change generated
     by the register displacement.
2) Nonlinear matched-collision experiment:
   - construct vertical-label and covariate corruptions with nearly identical
     final tangent predictive attribution;
   - show that the geometric register still separates their mechanisms;
   - quantify what the full trajectory adds over final scalar attribution.
3) Dynamic non-injectivity:
   - find probe pairs with nearly identical final register but substantially
     different register trajectories.

All randomness is seeded. Outputs are CSV/JSON/PDF in the script directory.
"""
from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.datasets import make_friedman1
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent

# -----------------------------------------------------------------------------
# Part I: exact linear factorization
# -----------------------------------------------------------------------------

def ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = X.shape[1]
    G = alpha * np.eye(p) + X.T @ X
    P = np.linalg.inv(G)
    theta = P @ (X.T @ y)
    return theta, P, G


def exact_linear_factorization(seed: int = 20260730, reps: int = 40, n_players: int = 8) -> Tuple[pd.DataFrame, dict]:
    """Exact Shapley values by subset enumeration, not Monte Carlo."""
    rng = np.random.default_rng(seed)
    records: List[dict] = []
    max_info_err = 0.0
    max_pred_err = 0.0
    nfact = math.factorial(n_players)
    weights = {
        k: math.factorial(k) * math.factorial(n_players-k-1) / nfact
        for k in range(n_players)
    }
    for rep in range(reps):
        p = 4
        n_anchor = 9
        n_val = 250
        alpha = 1.2
        beta = rng.normal(size=p)
        Xa = rng.normal(size=(n_anchor, p))
        ya = Xa @ beta + 0.35 * rng.normal(size=n_anchor)
        Xc = rng.normal(size=(n_players, p))
        yc = Xc @ beta + 0.35 * rng.normal(size=n_players)
        Xv = rng.normal(size=(n_val, p))
        yv = Xv @ beta + 0.35 * rng.normal(size=n_val)

        shap_info = np.zeros(n_players)
        shap_pred = np.zeros(n_players)
        reg_info = np.zeros(n_players)
        reg_pred = np.zeros(n_players)

        # Cache each coalition fit and utility.
        cache = {}
        for mask in range(1 << n_players):
            ids = [j for j in range(n_players) if (mask >> j) & 1]
            if ids:
                Xs = np.vstack([Xa, Xc[ids]])
                ys = np.r_[ya, yc[ids]]
            else:
                Xs, ys = Xa, ya
            theta, P, G = ridge_fit(Xs, ys, alpha)
            pred = Xv @ theta
            cache[mask] = {
                "theta": theta,
                "P": P,
                "G": G,
                "utility_pred": -float(np.mean((yv-pred)**2)),
                "utility_info": 0.5*float(np.linalg.slogdet(G)[1]),
                "pred": pred,
            }

        for mask in range(1 << n_players):
            k = int(mask.bit_count())
            if k == n_players:
                continue
            base = cache[mask]
            w = weights[k]
            for i in range(n_players):
                if (mask >> i) & 1:
                    continue
                nxt = cache[mask | (1 << i)]
                direct_info = nxt["utility_info"] - base["utility_info"]
                direct_pred = nxt["utility_pred"] - base["utility_pred"]

                # Register formulas at context S.
                x = Xc[i]; y = yc[i]
                q = float(x @ base["P"] @ x)
                nu = float(y - x @ base["theta"])
                d = (base["P"] @ x) * (nu / (1.0 + q))
                pred_new = base["pred"] + Xv @ d
                register_info = 0.5 * math.log1p(max(q, 0.0))
                register_pred = -float(np.mean((yv-pred_new)**2)) - base["utility_pred"]

                shap_info[i] += w * direct_info
                shap_pred[i] += w * direct_pred
                reg_info[i] += w * register_info
                reg_pred[i] += w * register_pred

        max_info_err = max(max_info_err, float(np.max(np.abs(shap_info - reg_info))))
        max_pred_err = max(max_pred_err, float(np.max(np.abs(shap_pred - reg_pred))))
        for i in range(n_players):
            records.append({
                "rep": rep,
                "player": i,
                "shapley_information": shap_info[i],
                "register_information_average": reg_info[i],
                "shapley_predictive": shap_pred[i],
                "register_predictive_average": reg_pred[i],
            })

    df = pd.DataFrame(records)
    summary = {
        "reps": reps,
        "n_players": n_players,
        "max_abs_error_information_factorization": max_info_err,
        "max_abs_error_predictive_factorization": max_pred_err,
        "spearman_information": float(spearmanr(df["shapley_information"], df["register_information_average"]).statistic),
        "spearman_predictive": float(spearmanr(df["shapley_predictive"], df["register_predictive_average"]).statistic),
    }
    return df, summary


# -----------------------------------------------------------------------------
# Part II: nonlinear matched collisions
# -----------------------------------------------------------------------------

@dataclass
class NLConfig:
    n_anchor: int = 500
    n_pool: int = 450
    n_test: int = 700
    d: int = 10
    h: int = 10
    noise: float = 1.0
    epochs: int = 220
    checkpoints: Tuple[int, ...] = (0, 15, 45, 100, 220)
    lr: float = 0.018
    weight_decay: float = 1e-4
    ridge_metric: float = 2e-2
    vertical_amp: float = 0.9
    n_pairs: int = 18
    reps: int = 6


class MLP(torch.nn.Module):
    def __init__(self, d: int, h: int):
        super().__init__()
        self.fc1 = torch.nn.Linear(d, h)
        self.fc2 = torch.nn.Linear(h, 1)
        torch.nn.init.normal_(self.fc1.weight, std=0.16)
        torch.nn.init.zeros_(self.fc1.bias)
        torch.nn.init.normal_(self.fc2.weight, std=0.16)
        torch.nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.tanh(self.fc1(x))).squeeze(-1)


def flatten(model: MLP) -> np.ndarray:
    with torch.no_grad():
        return np.concatenate([
            model.fc1.weight.detach().cpu().numpy().ravel(),
            model.fc1.bias.detach().cpu().numpy().ravel(),
            model.fc2.weight.detach().cpu().numpy().ravel(),
            model.fc2.bias.detach().cpu().numpy().ravel(),
        ])


def unpack(theta: np.ndarray, d: int, h: int):
    k = 0
    W = theta[k:k+h*d].reshape(h, d); k += h*d
    b = theta[k:k+h]; k += h
    a = theta[k:k+h]; k += h
    c = float(theta[k])
    return W, b, a, c


def forward_jac(theta: np.ndarray, X: np.ndarray, d: int, h: int) -> Tuple[np.ndarray, np.ndarray]:
    W, b, a, c = unpack(theta, d, h)
    Z = X @ W.T + b
    H = np.tanh(Z)
    sech2 = 1.0 - H * H
    pred = H @ a + c
    coeff = sech2 * a[None, :]
    JW = np.einsum("nh,nd->nhd", coeff, X).reshape(len(X), h*d)
    Jb = coeff
    Ja = H
    Jc = np.ones((len(X), 1))
    J = np.concatenate([JW, Jb, Ja, Jc], axis=1)
    return pred, J


def robust_sigma(r: np.ndarray) -> float:
    med = np.median(r)
    return float(max(1.4826 * np.median(np.abs(r - med)), 0.10))


def checkpoint_geometry(theta: np.ndarray, Xa: np.ndarray, ya: np.ndarray, Xt: np.ndarray, yt: np.ndarray, cfg: NLConfig):
    pa, Ja = forward_jac(theta, Xa, cfg.d, cfg.h)
    pt, Jt = forward_jac(theta, Xt, cfg.d, cfg.h)
    sigma = robust_sigma(pa - ya)
    p = Ja.shape[1]
    G = cfg.ridge_metric * np.eye(p) + (Ja.T @ Ja) / (sigma * sigma)
    P = np.linalg.inv(G)
    rt = pt - yt
    gtest = (Jt.T @ rt) / (sigma * sigma * len(Xt))
    return {"P": P, "sigma": sigma, "gtest": gtest, "Jtest": Jt, "test_resid": rt}


def point_register(theta: np.ndarray, x: np.ndarray, y: float, geom: dict, cfg: NLConfig) -> dict:
    pred, J = forward_jac(theta, x[None, :], cfg.d, cfg.h)
    j = J[0]
    r = float(pred[0] - y)
    P = geom["P"]
    sig = geom["sigma"]
    s = float(j @ P @ j) / (sig * sig)
    s = max(s, 0.0)
    c = s / (1.0 + s)
    z = r / sig
    effort = abs(z) * math.sqrt(s)
    dvec = -(P @ j) * (r / (sig * sig + float(j @ P @ j)))
    pred_influence = -float(geom["gtest"] @ dvec)  # first-order utility change on clean test loss
    function_impact = float(np.linalg.norm(geom["Jtest"] @ dvec) / math.sqrt(len(geom["Jtest"])))
    return {
        "contraction": c,
        "mu": s,
        "effort": effort,
        "residual_abs": abs(z),
        "pred_influence": pred_influence,
        "function_impact": function_impact,
        "dvec": dvec,
    }


def train_checkpoints(seed: int, cfg: NLConfig):
    X, y = make_friedman1(
        n_samples=cfg.n_anchor + cfg.n_pool + cfg.n_test,
        n_features=cfg.d,
        noise=cfg.noise,
        random_state=seed,
    )
    Xa = X[:cfg.n_anchor]; ya = y[:cfg.n_anchor]
    Xp = X[cfg.n_anchor:cfg.n_anchor+cfg.n_pool]; yp = y[cfg.n_anchor:cfg.n_anchor+cfg.n_pool]
    Xt = X[-cfg.n_test:]; yt = y[-cfg.n_test:]
    xs = StandardScaler().fit(Xa)
    Xa = xs.transform(Xa); Xp = xs.transform(Xp); Xt = xs.transform(Xt)
    ym, ys = float(ya.mean()), float(ya.std())
    ya = (ya-ym)/ys; yp=(yp-ym)/ys; yt=(yt-ym)/ys

    torch.manual_seed(seed)
    model = MLP(cfg.d, cfg.h).double()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    Xat = torch.from_numpy(Xa).double(); yat = torch.from_numpy(ya).double()
    thetas: Dict[int, np.ndarray] = {}
    for epoch in range(cfg.epochs+1):
        if epoch in cfg.checkpoints:
            thetas[epoch] = flatten(model)
        if epoch == cfg.epochs:
            break
        opt.zero_grad(set_to_none=True)
        loss = 0.5 * torch.mean((model(Xat)-yat)**2)
        loss.backward(); opt.step()
    geoms = {e: checkpoint_geometry(thetas[e], Xa, ya, Xt, yt, cfg) for e in cfg.checkpoints}
    return Xa, ya, Xp, yp, Xt, yt, thetas, geoms


def matched_collision_replicate(seed: int, cfg: NLConfig) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    Xa, ya, Xp, yp, Xt, yt, thetas, geoms = train_checkpoints(seed, cfg)
    final_e = cfg.checkpoints[-1]
    theta_f = thetas[final_e]; geom_f = geoms[final_e]

    # Base probes are selected independently of the register: random interior points
    # according to input norm (exclude only extreme 5% tails to avoid trivial leverage).
    norms = np.linalg.norm(Xp, axis=1)
    eligible = np.where(norms <= np.quantile(norms, 0.95))[0]
    bases = rng.choice(eligible, size=cfg.n_pairs, replace=False)

    grid = np.geomspace(0.08, 10.0, 34)
    rows: List[dict] = []
    pair_meta: List[dict] = []
    for pair_id, idx in enumerate(bases):
        x0 = Xp[idx].copy(); y0 = float(yp[idx])
        sign = float(rng.choice([-1.0, 1.0]))
        yv = y0 + sign * cfg.vertical_amp
        rv = point_register(theta_f, x0, yv, geom_f, cfg)
        target = rv["pred_influence"]

        best = None
        # Several random directions reduce accidental failure to find a matched scalar.
        for dir_id in range(10):
            direction = rng.normal(size=cfg.d)
            direction /= np.linalg.norm(direction) + 1e-12
            for amp in grid:
                xc = x0 + amp * direction
                rc = point_register(theta_f, xc, y0, geom_f, cfg)
                scale = max(abs(target), 2e-4)
                err = abs(rc["pred_influence"] - target) / scale
                # small regularizer avoids selecting huge amplitudes when ties occur
                crit = err + 2e-4 * amp
                if best is None or crit < best[0]:
                    best = (crit, err, amp, direction.copy(), rc)
        assert best is not None
        _, relerr, amp, direction, rc = best
        variants = {
            "label": (x0, yv),
            "covariate": (x0 + amp * direction, y0),
        }
        for typ, (xx, yy) in variants.items():
            for epoch in cfg.checkpoints:
                rr = point_register(thetas[epoch], xx, yy, geoms[epoch], cfg)
                rows.append({
                    "seed": seed,
                    "pair_id": pair_id,
                    "type": typ,
                    "epoch": epoch,
                    "contraction": rr["contraction"],
                    "effort": rr["effort"],
                    "residual_abs": rr["residual_abs"],
                    "pred_influence": rr["pred_influence"],
                    "function_impact": rr["function_impact"],
                    "covariate_amplitude": amp if typ == "covariate" else 0.0,
                })
        pair_meta.append({
            "seed": seed,
            "pair_id": pair_id,
            "target_pred_influence": target,
            "matched_pred_influence": rc["pred_influence"],
            "relative_match_error": relerr,
            "covariate_amplitude": amp,
            "label_amplitude": cfg.vertical_amp,
        })

    traj = pd.DataFrame(rows)
    meta = pd.DataFrame(pair_meta)

    # Dynamic non-injectivity on ordinary probes: same final (c,e), different paths.
    probe_ids = rng.choice(np.arange(len(Xp)), size=min(220, len(Xp)), replace=False)
    feats = []
    for idx in probe_ids:
        vals = []
        for epoch in cfg.checkpoints:
            rr = point_register(thetas[epoch], Xp[idx], float(yp[idx]), geoms[epoch], cfg)
            vals.extend([rr["contraction"], rr["effort"]])
        feats.append(vals)
    feats = np.asarray(feats)
    final = feats[:, -2:]
    # robust standardization for matching
    med = np.median(final, axis=0); scale = np.median(np.abs(final-med), axis=0)*1.4826 + 1e-8
    zf = (final-med)/scale
    path = feats[:, :-2]
    mp = np.median(path,axis=0); sp = np.median(np.abs(path-mp),axis=0)*1.4826 + 1e-8
    zp=(path-mp)/sp
    best_pair = None
    for i in range(len(probe_ids)):
        dist = np.linalg.norm(zf[i+1:] - zf[i], axis=1)
        if len(dist)==0: continue
        # restrict to close final states, then maximize historical separation
        candidates = np.where(dist <= np.quantile(dist, 0.08))[0]
        for cidx in candidates:
            j=i+1+cidx
            fd=float(np.linalg.norm(zf[i]-zf[j]))
            pdist=float(np.linalg.norm(zp[i]-zp[j]))
            score=pdist/(fd+0.05)
            if best_pair is None or score>best_pair[0]:
                best_pair=(score,i,j,fd,pdist)
    dyn = {}
    if best_pair is not None:
        _, i,j,fd,pdist=best_pair
        dyn={
            "seed":seed,
            "probe_i":int(probe_ids[i]),
            "probe_j":int(probe_ids[j]),
            "final_distance_standardized":fd,
            "trajectory_distance_standardized":pdist,
            "trajectory_i":feats[i].tolist(),
            "trajectory_j":feats[j].tolist(),
        }
    return traj, meta, dyn


def classification_summary(traj: pd.DataFrame, checkpoints: Tuple[int, ...]) -> dict:
    # Pivot one row per variant.
    wide_parts=[]
    for metric in ["contraction","effort","pred_influence","function_impact"]:
        p=traj.pivot_table(index=["seed","pair_id","type"],columns="epoch",values=metric)
        p.columns=[f"{metric}_e{int(c)}" for c in p.columns]
        wide_parts.append(p)
    wide=pd.concat(wide_parts,axis=1).reset_index()
    y=(wide["type"]=="covariate").astype(int).to_numpy()
    groups=wide["seed"].to_numpy()
    cv=GroupKFold(n_splits=len(np.unique(groups)))

    final=checkpoints[-1]
    feature_sets={
        "final_predictive_attribution_scalar":[f"pred_influence_e{final}"],
        "final_function_impact_scalar":[f"function_impact_e{final}"],
        "final_register_2d":[f"contraction_e{final}",f"effort_e{final}"],
        "dynamic_register":[f"{m}_e{e}" for e in checkpoints for m in ["contraction","effort"]],
        "dynamic_scalar_attribution":[f"pred_influence_e{e}" for e in checkpoints],
    }
    out={}
    for name,cols in feature_sets.items():
        X=wide[cols].to_numpy(float)
        # fold-safe standardization is not critical for AUC but use a simple global
        # robust transform; groups are independent replicates.
        med=np.median(X,axis=0); sc=np.median(np.abs(X-med),axis=0)*1.4826+1e-8
        X=(X-med)/sc
        model=LogisticRegression(C=1.0,max_iter=500,class_weight="balanced")
        pred=cross_val_predict(model,X,y,cv=cv,groups=groups,method="predict_proba")[:,1]
        out[name]={
            "auc":float(roc_auc_score(y,pred)),
            "n_features":len(cols),
        }
    return out,wide


def run_nonlinear(cfg: NLConfig) -> Tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame,dict]:
    trajs=[]; metas=[]; dyns=[]
    for r in range(cfg.reps):
        seed=9300+r
        print(f"nonlinear replicate {r+1}/{cfg.reps}",flush=True)
        t,m,d=matched_collision_replicate(seed,cfg)
        trajs.append(t);metas.append(m)
        if d: dyns.append(d)
    traj=pd.concat(trajs,ignore_index=True)
    meta=pd.concat(metas,ignore_index=True)
    dyn=pd.DataFrame(dyns)
    cls,wide=classification_summary(traj,cfg.checkpoints)
    final=traj[traj.epoch==cfg.checkpoints[-1]]
    label=final[final.type=="label"].sort_values(["seed","pair_id"])
    cov=final[final.type=="covariate"].sort_values(["seed","pair_id"])
    summary={
        "config":cfg.__dict__,
        "n_matched_pairs":int(len(meta)),
        "median_relative_match_error":float(meta.relative_match_error.median()),
        "q90_relative_match_error":float(meta.relative_match_error.quantile(.9)),
        "median_final_contraction_label":float(label.contraction.median()),
        "median_final_contraction_covariate":float(cov.contraction.median()),
        "median_final_effort_label":float(label.effort.median()),
        "median_final_effort_covariate":float(cov.effort.median()),
        "classification":cls,
        "dynamic_collision_median_final_distance":float(dyn.final_distance_standardized.median()) if len(dyn) else None,
        "dynamic_collision_median_trajectory_distance":float(dyn.trajectory_distance_standardized.median()) if len(dyn) else None,
    }
    return traj,meta,dyn,summary


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------

def make_plots(linear: pd.DataFrame, traj: pd.DataFrame, dyn: pd.DataFrame, cfg: NLConfig):
    fig,ax=plt.subplots(figsize=(5.5,4.5))
    ax.scatter(linear.shapley_information,linear.register_information_average,s=18,alpha=.55,label="information")
    lo=min(linear.shapley_information.min(),linear.register_information_average.min())
    hi=max(linear.shapley_information.max(),linear.register_information_average.max())
    ax.plot([lo,hi],[lo,hi],lw=1.2,ls="--")
    ax.set_xlabel("Shapley informationnel exact")
    ax.set_ylabel("Moyenne contextuelle du registre")
    ax.set_title("Factorisation exacte dans Ridge")
    ax.grid(alpha=.22); fig.tight_layout()
    fig.savefig(OUT/"factorization_exact_information.pdf"); plt.close(fig)

    final=traj[traj.epoch==cfg.checkpoints[-1]].copy()
    fig,ax=plt.subplots(figsize=(6.3,4.7))
    for typ,marker in [("label","o"),("covariate","s")]:
        s=final[final.type==typ]
        ax.scatter(s.contraction,s.effort,s=24,alpha=.58,marker=marker,label=typ)
    ax.set_xlabel("Contraction finale")
    ax.set_ylabel("Effort naturel final")
    ax.set_title("Même attribution prédictive, mécanismes différents")
    ax.legend(frameon=False); ax.grid(alpha=.22); fig.tight_layout()
    fig.savefig(OUT/"matched_scalar_collision_register.pdf"); plt.close(fig)

    # representative dynamic collision: median trajectory-distance row
    if len(dyn):
        row=dyn.iloc[(dyn.trajectory_distance_standardized-dyn.trajectory_distance_standardized.median()).abs().argmin()]
        ti=np.asarray(row.trajectory_i).reshape(len(cfg.checkpoints),2)
        tj=np.asarray(row.trajectory_j).reshape(len(cfg.checkpoints),2)
        fig,ax=plt.subplots(figsize=(6.5,4.5))
        ax.plot(cfg.checkpoints,ti[:,0],marker="o",label="point A - contraction")
        ax.plot(cfg.checkpoints,tj[:,0],marker="o",ls="--",label="point B - contraction")
        ax.plot(cfg.checkpoints,ti[:,1],marker="s",label="point A - effort")
        ax.plot(cfg.checkpoints,tj[:,1],marker="s",ls="--",label="point B - effort")
        ax.set_xlabel("Époque"); ax.set_ylabel("Valeur du registre")
        ax.set_title("États finaux proches, histoires géométriques différentes")
        ax.legend(frameon=False,ncol=2,fontsize=8); ax.grid(alpha=.22); fig.tight_layout()
        fig.savefig(OUT/"dynamic_noninjectivity_pair.pdf"); plt.close(fig)


def main():
    linear,linear_summary=exact_linear_factorization()
    linear.to_csv(OUT/"exact_factorization_linear.csv",index=False)

    cfg=NLConfig()
    traj,meta,dyn,nl_summary=run_nonlinear(cfg)
    traj.to_csv(OUT/"matched_collision_trajectories.csv",index=False)
    meta.to_csv(OUT/"matched_collision_pairs.csv",index=False)
    dyn.to_csv(OUT/"dynamic_noninjective_pairs.csv",index=False)
    make_plots(linear,traj,dyn,cfg)

    summary={"linear_factorization":linear_summary,"nonlinear_matched_collision":nl_summary}
    with open(OUT/"factorization_strictness_summary.json","w") as f:
        json.dump(summary,f,indent=2)
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
