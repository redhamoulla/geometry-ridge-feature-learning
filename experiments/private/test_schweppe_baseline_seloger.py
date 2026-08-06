#!/usr/bin/env python3
"""Direct comparison with a separately calibrated Schweppe-type GM baseline.

The script uses the same globally disjoint internal evaluation listing-ID pool as the BGR
stress test. It calls the exact same fit and scale functions that were used in
development calibration. Ridge/Huber/BGR rows are reused from
``seloger_bgr_results.csv`` under identical deterministic splits and
perturbations; only the Schweppe-type fits are recomputed.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import test_bounded_geometric_seloger as base  # noqa: E402

INPUT = Path(os.environ.get("SELOGER_CSV", "/mnt/data/selogerdata(1).csv"))
OUTDIR = Path(os.environ.get("EXPERIMENT_OUTPUT_DIR", str(HERE)))
OUTDIR.mkdir(parents=True, exist_ok=True)
DETAIL = OUTDIR / "seloger_schweppe_comparison.csv"
SUMMARY = OUTDIR / "seloger_schweppe_comparison_summary.json"
EXISTING = OUTDIR / "seloger_bgr_results.csv"


def rmse(model, X: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y[mask], model.predict(X)[mask])))


def main() -> None:
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)
    if not EXISTING.exists():
        raise FileNotFoundError(EXISTING)
    df_all = pd.read_csv(INPUT)
    for col in base.CAT_COLS:
        df_all[col] = df_all[col].astype(str)
    _dev_df, df, pool_metadata = base.development_evaluation_pools(df_all)

    schweppe_rows: list[dict] = []
    for split_no, seed in enumerate(base.EVAL_SEEDS, start=1):
        print(f"Schweppe split {split_no}/{len(base.EVAL_SEEDS)} seed={seed}", flush=True)
        train, val, test = base.split_data(df, seed)
        pre = base.make_preprocessor()
        Xtr = base.add_penalized_intercept(np.asarray(pre.fit_transform(train[base.FEATURE_COLS]), dtype=float))
        Xv = base.add_penalized_intercept(np.asarray(pre.transform(val[base.FEATURE_COLS]), dtype=float))
        Xte = base.add_penalized_intercept(np.asarray(pre.transform(test[base.FEATURE_COLS]), dtype=float))
        ytr = np.log(train["prix"].to_numpy(dtype=float))
        yv = np.log(val["prix"].to_numpy(dtype=float))
        yte = np.log(test["prix"].to_numpy(dtype=float))
        alpha = base.tune_alpha(Xtr, ytr, Xv, yv)
        q_geometry = float(base.ROBUST_HYPERPARAMETERS["schweppe_type"]["q_geometry"])
        kappa_y = float(base.ROBUST_HYPERPARAMETERS["schweppe_type"]["kappa_y"])
        kx = float(np.quantile(np.linalg.norm(Xtr, axis=1), q_geometry))
        scale = base.scale_for_geometry_budget(
            Xtr, ytr, train["idannonce"].to_numpy(), alpha, kx
        )
        plausible = (
            (test["surface"].to_numpy(dtype=float) > 0)
            & (test["nb_pieces"].to_numpy(dtype=float) <= 20)
            & (test["nb_chambres"].to_numpy(dtype=float) <= 10)
        )

        rng = np.random.default_rng(seed + 50000)
        master_ids = base.select_singletons(train, 0.20, rng)
        n_master = len(master_ids)
        x_cols = rng.integers(1, 1 + len(base.NUM_COLS), size=n_master)
        x_signs = rng.choice([-1.0, 1.0], size=n_master)
        y_signs = rng.choice([-1.0, 1.0], size=n_master)
        n5 = max(1, int(round(0.05 * train["idannonce"].nunique())))
        ids5 = master_ids[:n5]
        idx5 = base.row_indices_for_ids(train, ids5)
        cols5 = x_cols[: len(idx5)]
        xs5 = x_signs[: len(idx5)]
        ys5 = y_signs[: len(idx5)]

        scenarios: dict[str, tuple[float, np.ndarray, np.ndarray, np.ndarray]] = {}
        scenarios["nominal_raw"] = (0.0, Xtr, ytr, np.array([], dtype=int))
        yc = ytr.copy(); yc[idx5] += 4.0 * ys5
        scenarios["vertical_amplitude"] = (4.0, Xtr, yc, idx5)
        Xc = Xtr.copy(); Xc[idx5, cols5] += 50.0 * xs5
        scenarios["leverage_amplitude"] = (50.0, Xc, ytr, idx5)
        Xm = Xtr.copy(); ym = ytr.copy(); Xm[idx5, cols5] += 50.0 * xs5; ym[idx5] += 4.0 * ys5
        scenarios["mixed_amplitude"] = (50.0, Xm, ym, idx5)
        noise = base.group_noise_vector(train, np.random.default_rng(seed + 60000))
        scenarios["gaussian_noise"] = (0.4, Xtr, ytr + 0.4 * noise, np.arange(len(ytr)))

        nominal_fit = base.fit_schweppe(
            Xtr, ytr, alpha, kx, kappa_y, scale,
            max_iter=base.SCHWEPPE_MAX_IRLS, tol=base.SCHWEPPE_TOL,
        )
        all_mask = np.ones(len(yte), dtype=bool)
        nominal_rmse_plausible = rmse(nominal_fit.model, Xte, yte, plausible)
        nominal_rmse_all = rmse(nominal_fit.model, Xte, yte, all_mask)
        for scenario, (level, Xs, ys, corrupt_idx) in scenarios.items():
            fitted = nominal_fit if scenario == "nominal_raw" else base.fit_schweppe(
                Xs, ys, alpha, kx, kappa_y, scale,
                max_iter=base.SCHWEPPE_MAX_IRLS, tol=base.SCHWEPPE_TOL,
            )
            r_plausible = rmse(fitted.model, Xte, yte, plausible)
            r_all = rmse(fitted.model, Xte, yte, all_mask)
            schweppe_rows.append({
                "seed": seed,
                "scenario": scenario,
                "level": level,
                "method": "schweppe_type",
                "rmse": r_plausible,
                "delta_rmse": r_plausible - nominal_rmse_plausible,
                "rmse_all": r_all,
                "delta_rmse_all": r_all - nominal_rmse_all,
                "mean_corrupt_weight": float(np.mean(fitted.weights[corrupt_idx])) if len(corrupt_idx) else math.nan,
                "mean_corrupt_leverage_factor": float(np.mean(fitted.leverage_factor[corrupt_idx])) if len(corrupt_idx) else math.nan,
                "irls_iterations": fitted.n_iter,
                "alpha": alpha,
                "kappa_x": kx,
                "q_geometry": q_geometry,
                "kappa_y": kappa_y,
                "scale": scale,
            })

    new = pd.DataFrame(schweppe_rows)
    old = pd.read_csv(EXISTING)
    keep = (
        ((old.scenario == "nominal_raw") & np.isclose(old.level, 0.0))
        | ((old.scenario == "vertical_amplitude") & np.isclose(old.level, 4.0))
        | ((old.scenario == "leverage_amplitude") & np.isclose(old.level, 50.0))
        | ((old.scenario == "mixed_amplitude") & np.isclose(old.level, 50.0))
        | ((old.scenario == "gaussian_noise") & np.isclose(old.level, 0.4))
    ) & old.method.isin(["ridge", "huber", "geometry", "bgr"])
    existing = old.loc[keep, ["seed", "scenario", "level", "method", "rmse_plausible", "delta_rmse_plausible", "rmse_all", "delta_rmse_all", "mean_corrupt_combined_weight"]].copy()
    existing = existing.rename(columns={
        "rmse_plausible": "rmse",
        "delta_rmse_plausible": "delta_rmse",
        "mean_corrupt_combined_weight": "mean_corrupt_weight",
    })
    for column in ["mean_corrupt_leverage_factor", "irls_iterations", "alpha", "kappa_x", "q_geometry", "kappa_y", "scale"]:
        existing[column] = math.nan
    detail = pd.concat([existing, new], ignore_index=True, sort=False)
    detail.to_csv(DETAIL, index=False)
    agg = (detail.groupby(["scenario", "level", "method"], as_index=False)
           .agg(n=("seed", "size"), rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
                rmse_all_mean=("rmse_all", "mean"), rmse_all_std=("rmse_all", "std"),
                delta_rmse_mean=("delta_rmse", "mean"), delta_rmse_std=("delta_rmse", "std"),
                delta_rmse_all_mean=("delta_rmse_all", "mean"), delta_rmse_all_std=("delta_rmse_all", "std"),
                corrupt_weight_mean=("mean_corrupt_weight", "mean")))
    payload = {
        "protocol": {
            "global_dev_eval_partition": pool_metadata,
            "splits": base.EVAL_SEEDS,
            "methods": ["ridge", "huber", "geometry", "schweppe_type", "bgr"],
            "formula": "w_i=min(1,kappa_y v_i/|u_i|), v_i=min(1,kappa_x/||x_i||)",
            "warning": "Declared Schweppe-type GM baseline using the same feature-norm leverage proxy as BGR but separately calibrated; not a high-breakdown Krasker-Welsch implementation.",
            "hyperparameter_source": base.HYPERPARAM_PATH.name,
            "penalized_intercept": True,
            "same_fit_functions_calibration_evaluation": True,
            "reuse": "Ridge/Huber/BGR rows are read from seloger_bgr_results.csv under identical deterministic splits and perturbations; only Schweppe-type fits are recomputed.",
        },
        "summary": agg.to_dict(orient="records"),
    }
    SUMMARY.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # Final five-method figure used in the manuscript.
    scenario_specs = [
        ("vertical_amplitude", 4.0, "Labels"),
        ("leverage_amplitude", 50.0, "Leverage"),
        ("mixed_amplitude", 50.0, "Mixte"),
        ("gaussian_noise", 0.4, "Gaussien"),
    ]
    methods = ["ridge", "huber", "geometry", "schweppe_type", "bgr"]
    labels = {"ridge": "Ridge", "huber": "Huber", "geometry": "Géométrie",
              "schweppe_type": "Schweppe", "bgr": "BGR"}
    x = np.arange(len(scenario_specs), dtype=float)
    width = 0.155
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    for j, method in enumerate(methods):
        values = []
        for scenario, level, _label in scenario_specs:
            row = agg[(agg.scenario == scenario) & np.isclose(agg.level, level) & (agg.method == method)]
            values.append(float(row.iloc[0].delta_rmse_mean))
        ax.bar(x + (j - 2) * width, values, width=width, label=labels[method])
    ax.axhline(0, linewidth=0.8)
    ax.set_xticks(x, [label for _, _, label in scenario_specs])
    ax.set_ylabel("Augmentation de RMSE plausible")
    ax.set_title("SeLoger : dégradation sous contamination — pool interne d’évaluation")
    ax.legend(ncol=3, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUTDIR / "seloger_bgr_core_scenarios.png", dpi=220)
    fig.savefig(OUTDIR / "seloger_bgr_core_scenarios.pdf")
    plt.close(fig)

    print(agg.to_string(index=False))
    print(f"wrote {DETAIL}")
    print(f"wrote {SUMMARY}")


if __name__ == "__main__":
    main()
