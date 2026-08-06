#!/usr/bin/env python3
"""Development-only calibration of BGR and a Schweppe-type baseline on SeLoger.

A single listing-ID partition is reserved before any model selection:

    D_dev ∩ D_eval = ∅.

All robust thresholds are calibrated only inside D_dev. The ten final repeated
splits are later performed only inside D_eval. Calibration and evaluation call
the same fitting and scale functions, with identical iteration limits and
tolerances.

For each robust family, a predeclared ladder of increasingly weak clipping
settings is scanned from strongest to weakest. We select the most aggressive
candidate whose *mean of per-split nominal RMSE ratios* does not exceed 1.005.
The maximum per-split ratio is reported but is not the selection constraint.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import test_bounded_geometric_seloger as base  # noqa: E402

INPUT = Path(os.environ.get("SELOGER_CSV", "/mnt/data/selogerdata(1).csv"))
OUTDIR = Path(os.environ.get("EXPERIMENT_OUTPUT_DIR", str(HERE)))
OUTDIR.mkdir(parents=True, exist_ok=True)
TABLE_PATH = OUTDIR / "seloger_robust_hyperparameter_development.csv"
JSON_PATH = OUTDIR / "seloger_robust_hyperparameters.json"
PARTITION_PATH = OUTDIR / "seloger_dev_eval_partition.json"

DEV_SEEDS = [3101, 3202, 3303]
CANDIDATE_LADDER = [
    (0.90, 2.0),
    (0.95, 2.0),
    (0.95, 3.0),
    (0.975, 3.0),
    (0.99, 4.0),
    (1.00, 6.0),
]
NOMINAL_RATIO_LIMIT = 1.005


def rmse(model, X: np.ndarray, y: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y, model.predict(X))))


def fit_family(
    family: str,
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    kappa_x: float,
    kappa_y: float,
    scale: float,
):
    if family == "bgr":
        return base.fit_method(
            "bgr", X, y, alpha, kappa_x, kappa_y, scale,
            tol=base.BGR_TOL, max_iter=base.BGR_MAX_IRLS,
        ).model
    if family == "schweppe_type":
        return base.fit_schweppe(
            X, y, alpha, kappa_x, kappa_y, scale,
            tol=base.SCHWEPPE_TOL, max_iter=base.SCHWEPPE_MAX_IRLS,
        ).model
    raise ValueError(family)


def main() -> None:
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)
    df_all = pd.read_csv(INPUT)
    for col in base.CAT_COLS:
        df_all[col] = df_all[col].astype(str)
    dev_df, _eval_df, pool_metadata = base.development_evaluation_pools(df_all)
    PARTITION_PATH.write_text(
        json.dumps(pool_metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    rows: list[dict] = []
    for seed in DEV_SEEDS:
        print(f"development seed={seed}", flush=True)
        train, val, _ = base.split_data(dev_df, seed)
        pre = base.make_preprocessor()
        Xtr = base.add_penalized_intercept(
            np.asarray(pre.fit_transform(train[base.FEATURE_COLS]), dtype=float)
        )
        Xv = base.add_penalized_intercept(
            np.asarray(pre.transform(val[base.FEATURE_COLS]), dtype=float)
        )
        ytr = np.log(train["prix"].to_numpy(dtype=float))
        yv = np.log(val["prix"].to_numpy(dtype=float))
        groups = train["idannonce"].to_numpy()
        alpha = base.tune_alpha(Xtr, ytr, Xv, yv)

        ridge = base.make_ridge(alpha)
        ridge.fit(Xtr, ytr)
        ridge_rmse = rmse(ridge, Xv, yv)
        norms = np.linalg.norm(Xtr, axis=1)

        for family in ["bgr", "schweppe_type"]:
            for rank, (q, kappa_y) in enumerate(CANDIDATE_LADDER):
                kappa_x = float(np.quantile(norms, q))
                # This is exactly the scale construction used at evaluation.
                scale = base.scale_for_geometry_budget(
                    Xtr, ytr, groups, alpha, kappa_x
                )
                model = fit_family(family, Xtr, ytr, alpha, kappa_x, kappa_y, scale)
                nominal_rmse = rmse(model, Xv, yv)
                rows.append({
                    "seed": seed,
                    "family": family,
                    "aggressiveness_rank": rank,
                    "q_geometry": q,
                    "kappa_y": kappa_y,
                    "alpha": alpha,
                    "kappa_x": kappa_x,
                    "scale": scale,
                    "ridge_nominal_rmse": ridge_rmse,
                    "nominal_rmse": nominal_rmse,
                    "nominal_ratio": nominal_rmse / ridge_rmse,
                    "bgr_max_iter": base.BGR_MAX_IRLS,
                    "bgr_tol": base.BGR_TOL,
                    "schweppe_max_iter": base.SCHWEPPE_MAX_IRLS,
                    "schweppe_tol": base.SCHWEPPE_TOL,
                })

    detail = pd.DataFrame(rows)
    detail.to_csv(TABLE_PATH, index=False)
    aggregate = (
        detail.groupby(
            ["family", "aggressiveness_rank", "q_geometry", "kappa_y"],
            as_index=False,
        )
        .agg(
            n=("seed", "size"),
            nominal_ratio_mean=("nominal_ratio", "mean"),
            nominal_ratio_max=("nominal_ratio", "max"),
            nominal_rmse_mean=("nominal_rmse", "mean"),
            ridge_nominal_rmse_mean=("ridge_nominal_rmse", "mean"),
        )
        .sort_values(["family", "aggressiveness_rank"])
    )

    selected: dict[str, dict] = {}
    for family in ["bgr", "schweppe_type"]:
        candidates = aggregate[aggregate.family == family].copy()
        eligible = candidates[candidates.nominal_ratio_mean <= NOMINAL_RATIO_LIMIT]
        if eligible.empty:
            best = candidates.sort_values("nominal_ratio_mean").iloc[0]
            status = "fallback_best_nominal"
        else:
            best = eligible.sort_values("aggressiveness_rank").iloc[0]
            status = "most_aggressive_under_mean_of_ratios_constraint"
        selected[family] = {
            "q_geometry": float(best.q_geometry),
            "kappa_y": float(best.kappa_y),
            "aggressiveness_rank": int(best.aggressiveness_rank),
            "nominal_ratio_mean": float(best.nominal_ratio_mean),
            "nominal_ratio_max": float(best.nominal_ratio_max),
            "nominal_rmse_mean": float(best.nominal_rmse_mean),
            "ridge_nominal_rmse_mean": float(best.ridge_nominal_rmse_mean),
            "selection_status": status,
        }

    payload = {
        "protocol": {
            "global_dev_eval_partition": pool_metadata,
            "development_seeds": DEV_SEEDS,
            "evaluation_seeds": base.EVAL_SEEDS,
            "candidate_ladder": [
                {"q_geometry": q, "kappa_y": k} for q, k in CANDIDATE_LADDER
            ],
            "nominal_ratio_limit": NOMINAL_RATIO_LIMIT,
            "nominal_constraint_definition": "arithmetic mean across development splits of (robust validation RMSE / Ridge validation RMSE) <= 1.005",
            "selection_rule": "choose separately for each family the most aggressive eligible candidate; report the maximum splitwise ratio but do not use it as the constraint",
            "target": "log(price)",
            "penalized_intercept": True,
            "same_fit_functions_calibration_evaluation": True,
            "bgr_max_iter": base.BGR_MAX_IRLS,
            "bgr_tol": base.BGR_TOL,
            "schweppe_max_iter": base.SCHWEPPE_MAX_IRLS,
            "schweppe_tol": base.SCHWEPPE_TOL,
        },
        "bgr": selected["bgr"],
        "schweppe_type": selected["schweppe_type"],
    }
    JSON_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(aggregate.to_string(index=False))
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"wrote {TABLE_PATH}")
    print(f"wrote {JSON_PATH}")
    print(f"wrote {PARTITION_PATH}")


if __name__ == "__main__":
    main()
