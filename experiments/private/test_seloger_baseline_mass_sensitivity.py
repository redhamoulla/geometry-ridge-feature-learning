#!/usr/bin/env python3
"""SeLoger sensitivity checks requested by the V6 review.

This script addresses three narrow questions without changing the main stress
protocol:

1. compare the former Huber ablation, whose residual scale comes from a
   geometry-weighted prefit, with a pure Huber baseline whose scale comes from
   ordinary Ridge;
2. compare three calibrations of the geometric threshold kappa_x: raw-line
   quantile, one statistical mass per listing ID, and a row-weighted quantile
   with weights 1 / multiplicity;
3. audit the deterministic representative used for validation/test when raw
   rows sharing an idannonce are not exactly identical.

The training table remains raw and pi_i = chi_i = 1 in all fitted estimators.
Only threshold calibration is varied in item 2.
"""
from __future__ import annotations

import json
import math
import os
from collections import Counter
from pathlib import Path
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import test_bounded_geometric_seloger as base  # noqa: E402

INPUT = Path(os.environ.get("SELOGER_CSV", "/mnt/data/selogerdata(1).csv"))
OUTDIR = Path(os.environ.get("EXPERIMENT_OUTPUT_DIR", str(HERE)))
OUTDIR.mkdir(parents=True, exist_ok=True)

DETAIL_PATH = OUTDIR / "seloger_baseline_mass_sensitivity_results.csv"
SUMMARY_PATH = OUTDIR / "seloger_baseline_mass_sensitivity_summary.csv"
JSON_PATH = OUTDIR / "seloger_baseline_mass_sensitivity_summary.json"
HUBER_TUNING_PATH = OUTDIR / "seloger_pure_huber_tuning.csv"
HUBER_HYPER_PATH = OUTDIR / "seloger_pure_huber_hyperparameters.json"
AUDIT_PATH = OUTDIR / "seloger_listing_representative_audit.json"
FIGURE_PATH = OUTDIR / "seloger_baseline_mass_sensitivity.pdf"

DEV_SEEDS = [3101, 3202, 3303]
HUBER_KY_LADDER = [1.5, 2.0, 3.0, 4.0, 6.0]
NOMINAL_RATIO_LIMIT = 1.005
Q_GEOMETRY = base.Q_GEOMETRY
EVAL_SEEDS = base.EVAL_SEEDS[:5]


def rmse(model, X: np.ndarray, y: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y, model.predict(X))))


def ordinary_ridge_scale(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    alpha: float,
) -> float:
    model = base.make_ridge(alpha)
    model.fit(X, y)
    return base.robust_scale_from_unique_groups(model, X, y, groups)


def weighted_quantile(values: np.ndarray, q: float, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(mask):
        raise ValueError("No positive finite weight")
    order = np.argsort(values[mask], kind="mergesort")
    v = values[mask][order]
    w = weights[mask][order]
    c = np.cumsum(w)
    target = float(q) * float(c[-1])
    return float(v[min(np.searchsorted(c, target, side="left"), len(v) - 1)])


def kappa_x_schemes(norms: np.ndarray, groups: np.ndarray, q: float) -> dict[str, float]:
    groups = np.asarray(groups)
    counts = pd.Series(groups).value_counts().to_dict()
    line = float(np.quantile(norms, q))
    medians = pd.DataFrame({"group": groups, "norm": norms}).groupby("group", sort=True).norm.median().to_numpy()
    per_id = float(np.quantile(medians, q))
    weights = np.array([1.0 / counts[g] for g in groups], dtype=float)
    pi_weighted = weighted_quantile(norms, q, weights)
    return {"line": line, "per_id": per_id, "pi_weighted": pi_weighted}


def prepare_split(frame: pd.DataFrame, seed: int):
    train, val, test = base.split_data(frame, seed)
    pre = base.make_preprocessor()
    Xtr = base.add_penalized_intercept(np.asarray(pre.fit_transform(train[base.FEATURE_COLS]), dtype=float))
    Xv = base.add_penalized_intercept(np.asarray(pre.transform(val[base.FEATURE_COLS]), dtype=float))
    Xte = base.add_penalized_intercept(np.asarray(pre.transform(test[base.FEATURE_COLS]), dtype=float))
    ytr = np.log(train["prix"].to_numpy(dtype=float))
    yv = np.log(val["prix"].to_numpy(dtype=float))
    yte = np.log(test["prix"].to_numpy(dtype=float))
    groups = train["idannonce"].to_numpy()
    alpha = base.tune_alpha(Xtr, ytr, Xv, yv)
    plausible = (
        (test["surface"].to_numpy(dtype=float) > 0)
        & (test["nb_pieces"].to_numpy(dtype=float) <= 20)
        & (test["nb_chambres"].to_numpy(dtype=float) <= 10)
    )
    return train, Xtr, Xv, Xte, ytr, yv, yte, groups, alpha, plausible


def calibrate_pure_huber(dev_df: pd.DataFrame) -> float:
    rows: list[dict] = []
    for seed in DEV_SEEDS:
        train, Xtr, Xv, _Xte, ytr, yv, _yte, groups, alpha, _plausible = prepare_split(dev_df, seed)
        scale = ordinary_ridge_scale(Xtr, ytr, groups, alpha)
        ridge = base.make_ridge(alpha)
        ridge.fit(Xtr, ytr)
        base_rmse = rmse(ridge, Xv, yv)
        for rank, ky in enumerate(HUBER_KY_LADDER):
            fit = base.fit_method("huber", Xtr, ytr, alpha, math.inf, ky, scale)
            value = rmse(fit.model, Xv, yv)
            rows.append({
                "seed": seed,
                "rank": rank,
                "kappa_y": ky,
                "ridge_rmse": base_rmse,
                "huber_rmse": value,
                "nominal_ratio": value / base_rmse,
                "scale": scale,
            })
    detail = pd.DataFrame(rows)
    detail.to_csv(HUBER_TUNING_PATH, index=False)
    agg = detail.groupby(["rank", "kappa_y"], as_index=False).nominal_ratio.agg(["mean", "max"]).reset_index()
    eligible = agg[agg["mean"] <= NOMINAL_RATIO_LIMIT]
    if eligible.empty:
        selected = agg.sort_values(["mean", "max"]).iloc[0]
        status = "fallback_best_nominal"
    else:
        selected = eligible.sort_values("rank").iloc[0]
        status = "most_aggressive_under_mean_ratio_constraint"
    payload = {
        "development_seeds": DEV_SEEDS,
        "scale": "ordinary Ridge residual MAD on one representative per listing",
        "nominal_ratio_limit": NOMINAL_RATIO_LIMIT,
        "selected_kappa_y": float(selected.kappa_y),
        "nominal_ratio_mean": float(selected["mean"]),
        "nominal_ratio_max": float(selected["max"]),
        "status": status,
    }
    HUBER_HYPER_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return float(selected.kappa_y)


def listing_audit(df: pd.DataFrame) -> dict:
    multiplicity = df.groupby("idannonce", sort=False).size()
    divergent: list[dict] = []
    patterns: Counter[tuple[str, ...]] = Counter()
    for listing_id, group in df.groupby("idannonce", sort=False):
        if len(group) <= 1 or len(group.drop_duplicates()) == 1:
            continue
        columns = tuple(c for c in df.columns if group[c].nunique(dropna=False) > 1)
        patterns[columns] += 1
        divergent.append({"idannonce": int(listing_id), "multiplicity": int(len(group)), "differing_columns": list(columns)})
    return {
        "rows": int(len(df)),
        "unique_listings": int(df.idannonce.nunique()),
        "repeated_listing_groups": int((multiplicity > 1).sum()),
        "extra_rows_beyond_first": int((multiplicity - 1).clip(lower=0).sum()),
        "maximum_multiplicity": int(multiplicity.max()),
        "nonidentical_repeated_groups": int(len(divergent)),
        "nonidentical_fraction_of_all_listings": float(len(divergent) / df.idannonce.nunique()),
        "differing_column_patterns": {"|".join(k): int(v) for k, v in patterns.items()},
        "representative_rule": "numeric median and lexicographically first modal non-numeric value, computed per idannonce; independent of row order",
        "divergent_groups": divergent,
    }


def corruptions(train: pd.DataFrame, Xtr: np.ndarray, ytr: np.ndarray, seed: int):
    rng = np.random.default_rng(seed + 50000)
    master_ids = base.select_singletons(train, 0.20, rng)
    n5 = max(1, int(round(0.05 * train["idannonce"].nunique())))
    ids = master_ids[:n5]
    idx = base.row_indices_for_ids(train, ids)
    cols = rng.integers(1, 1 + len(base.NUM_COLS), size=len(master_ids))[:len(idx)]
    xs = rng.choice([-1.0, 1.0], size=len(master_ids))[:len(idx)]
    ys = rng.choice([-1.0, 1.0], size=len(master_ids))[:len(idx)]
    yc = ytr.copy(); yc[idx] += 4.0 * ys
    xl = Xtr.copy(); xl[idx, cols] += 50.0 * xs
    xm = Xtr.copy(); ym = ytr.copy(); xm[idx, cols] += 50.0 * xs; ym[idx] += 4.0 * ys
    return idx, {"nominal": (Xtr, ytr), "vertical": (Xtr, yc), "leverage": (xl, ytr), "mixed": (xm, ym)}


def evaluate(eval_df: pd.DataFrame, pure_ky: float) -> pd.DataFrame:
    rows: list[dict] = []
    for seed in EVAL_SEEDS:
        train, Xtr, Xv, Xte, ytr, yv, yte, groups, alpha, plausible = prepare_split(eval_df, seed)
        norms = np.linalg.norm(Xtr, axis=1)
        thresholds = kappa_x_schemes(norms, groups, Q_GEOMETRY)
        pure_scale = ordinary_ridge_scale(Xtr, ytr, groups, alpha)
        anchored_scale = base.scale_for_geometry_budget(Xtr, ytr, groups, alpha, thresholds["line"])
        idx, scenarios = corruptions(train, Xtr, ytr, seed)

        # The Huber comparison isolates the scale construction.
        for label, scale, ky in [
            ("huber_pure", pure_scale, pure_ky),
            ("huber_geometry_anchored_scale", anchored_scale, base.KAPPA_Y),
        ]:
            nominal_fit = base.fit_method("huber", Xtr, ytr, alpha, math.inf, ky, scale)
            nominal_rmse = rmse(nominal_fit.model, Xte[plausible], yte[plausible])
            for scenario in ["nominal", "vertical"]:
                xx, yy = scenarios[scenario]
                fit = nominal_fit if scenario == "nominal" else base.fit_method("huber", xx, yy, alpha, math.inf, ky, scale)
                value = rmse(fit.model, Xte[plausible], yte[plausible])
                rows.append({
                    "seed": seed, "family": "huber_scale", "variant": label,
                    "scenario": scenario, "rmse": value, "delta_rmse": value - nominal_rmse,
                    "kappa_x": math.inf, "kappa_y": ky, "scale": scale,
                    "mean_corrupt_weight": float(np.mean(fit.combined_weights[idx])) if scenario != "nominal" else math.nan,
                })

        # Sensitivity of kappa_x only; pi_i=chi_i=1 in all fits.
        for scheme, kx in thresholds.items():
            scale = base.scale_for_geometry_budget(Xtr, ytr, groups, alpha, kx)
            nominal_fit = base.fit_method("bgr", Xtr, ytr, alpha, kx, base.KAPPA_Y, scale)
            nominal_rmse = rmse(nominal_fit.model, Xte[plausible], yte[plausible])
            for scenario in ["nominal", "vertical", "leverage", "mixed"]:
                xx, yy = scenarios[scenario]
                fit = nominal_fit if scenario == "nominal" else base.fit_method("bgr", xx, yy, alpha, kx, base.KAPPA_Y, scale)
                value = rmse(fit.model, Xte[plausible], yte[plausible])
                rows.append({
                    "seed": seed, "family": "kappa_x", "variant": scheme,
                    "scenario": scenario, "rmse": value, "delta_rmse": value - nominal_rmse,
                    "kappa_x": kx, "kappa_y": base.KAPPA_Y, "scale": scale,
                    "mean_corrupt_weight": float(np.mean(fit.combined_weights[idx])) if scenario != "nominal" else math.nan,
                })
    return pd.DataFrame(rows)


def main() -> None:
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)
    df_all = pd.read_csv(INPUT)
    for col in base.CAT_COLS:
        df_all[col] = df_all[col].astype(str)
    audit = listing_audit(df_all)
    AUDIT_PATH.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    dev_df, eval_df, pool = base.development_evaluation_pools(df_all)
    if HUBER_HYPER_PATH.exists() and HUBER_TUNING_PATH.exists():
        pure_payload = json.loads(HUBER_HYPER_PATH.read_text(encoding="utf-8"))
        pure_ky = float(pure_payload["selected_kappa_y"])
    else:
        pure_ky = calibrate_pure_huber(dev_df)
    detail = evaluate(eval_df, pure_ky)
    detail.to_csv(DETAIL_PATH, index=False)
    summary = detail.groupby(["family", "variant", "scenario"], as_index=False).agg(
        n=("seed", "size"),
        rmse_mean=("rmse", "mean"),
        rmse_std=("rmse", "std"),
        delta_rmse_mean=("delta_rmse", "mean"),
        delta_rmse_std=("delta_rmse", "std"),
        kappa_x_mean=("kappa_x", "mean"),
        scale_mean=("scale", "mean"),
        mean_corrupt_weight=("mean_corrupt_weight", "mean"),
    )
    summary.to_csv(SUMMARY_PATH, index=False)
    payload = {
        "protocol": {
            "pool": pool,
            "evaluation_seeds": EVAL_SEEDS,
            "q_geometry": Q_GEOMETRY,
            "pi_i": 1.0,
            "chi_i": 1.0,
            "note": "multiplicity is used only to define alternative kappa_x calibration distributions; it is not used as a fitting mass",
        },
        "pure_huber": json.loads(HUBER_HYPER_PATH.read_text(encoding="utf-8")),
        "listing_representative_audit": audit,
        "summary": summary.to_dict(orient="records"),
    }
    JSON_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # Compact two-panel figure.
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))
    left = summary[(summary.family == "huber_scale") & (summary.scenario == "vertical")]
    variants_h = ["huber_pure", "huber_geometry_anchored_scale"]
    vals_h = [float(left[left.variant == v].delta_rmse_mean.iloc[0]) for v in variants_h]
    axes[0].bar(np.arange(2), vals_h)
    axes[0].set_xticks(np.arange(2), ["Pure", "Échelle géométrique"]); axes[0].set_ylabel("Augmentation de RMSE"); axes[0].set_title("Huber sous aberration verticale")

    right = summary[(summary.family == "kappa_x") & (summary.scenario.isin(["leverage", "mixed"]))]
    scen2 = ["leverage", "mixed"]; labels2 = ["Leverage", "Mixte"]; x2 = np.arange(2); width2 = .25
    for j, variant in enumerate(["line", "per_id", "pi_weighted"]):
        vals = [float(right[(right.variant == variant) & (right.scenario == s)].delta_rmse_mean.iloc[0]) for s in scen2]
        axes[1].bar(x2 + (j - 1) * width2, vals, width=width2, label=variant)
    axes[1].set_xticks(x2, labels2); axes[1].set_ylabel("Augmentation de RMSE"); axes[1].set_title(r"Sensibilité de $\kappa_X$"); axes[1].legend(fontsize=8)
    for ax in axes: ax.axhline(0, linewidth=.8)
    fig.tight_layout(); fig.savefig(FIGURE_PATH); fig.savefig(OUTDIR / "seloger_baseline_mass_sensitivity.png", dpi=220); plt.close(fig)
    print(json.dumps(payload["pure_huber"], ensure_ascii=False, indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
