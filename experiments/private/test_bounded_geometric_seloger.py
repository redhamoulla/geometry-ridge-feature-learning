#!/usr/bin/env python3
"""Stress test of bounded-contraction / bounded-innovation Ridge on raw SeLoger data.

The training table is kept raw: exact repeated rows, malformed covariates and all
other natural irregularities remain present. Splits are made by idannonce to
avoid train/test leakage. Test risk is reported per unique listing.

Methods
-------
- ridge: ordinary Ridge.
- huber: bounded innovation only (Huber residual score).
- geometry: bounded geometric contribution only (row-norm budget).
- bgr: both budgets.

The geometric budget is computed in the feature coordinates produced by a
RobustScaler + one-hot encoder, augmented with an explicit constant coordinate.
The intercept is therefore penalized exactly like the other directions. The
robust thresholds are loaded from a development protocol declared in
``seloger_robust_hyperparameters.json`` and are frozen before evaluation.
"""
from __future__ import annotations

import json
import math
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder, RobustScaler

warnings.filterwarnings("ignore")

INPUT = Path(os.environ.get("SELOGER_CSV", "/mnt/data/selogerdata(1).csv"))
OUTDIR = Path(os.environ.get("EXPERIMENT_OUTPUT_DIR", str(Path(__file__).resolve().parent)))
OUTDIR.mkdir(parents=True, exist_ok=True)
DETAIL_PATH = OUTDIR / "seloger_bgr_results.csv"
SUMMARY_PATH = OUTDIR / "seloger_bgr_summary.csv"
PAIR_PATH = OUTDIR / "seloger_bgr_pairwise.csv"
JSON_PATH = OUTDIR / "seloger_bgr_summary.json"

NUM_COLS = ["surface", "nb_pieces", "nb_chambres", "nb_photos", "si_balcon", "position"]
CAT_COLS = [
    "codepostal",
    "typedebien",
    "idtypecuisine",
    "idtypepublicationsourcecouplage",
    "idagence",
]
FEATURE_COLS = NUM_COLS + CAT_COLS
ALPHAS = np.logspace(-3, 4, 15)
METHODS = ["ridge", "huber", "geometry", "bgr"]
EVAL_SEEDS = [1101, 1202, 1303, 1404, 1505, 1606, 1707, 1808, 1909, 2010]
HYPERPARAM_PATH = OUTDIR / "seloger_robust_hyperparameters.json"


def load_robust_hyperparameters() -> dict:
    defaults = {
        "bgr": {"q_geometry": 0.95, "kappa_y": 3.0},
        "schweppe_type": {"q_geometry": 0.95, "kappa_y": 3.0},
        "selection_status": "fallback_defaults",
    }
    if not HYPERPARAM_PATH.exists():
        return defaults
    payload = json.loads(HYPERPARAM_PATH.read_text(encoding="utf-8"))
    for key in ("bgr", "schweppe_type"):
        if key not in payload:
            raise ValueError(f"Missing {key} in {HYPERPARAM_PATH}")
    return payload


ROBUST_HYPERPARAMETERS = load_robust_hyperparameters()
Q_GEOMETRY = float(ROBUST_HYPERPARAMETERS["bgr"]["q_geometry"])
KAPPA_Y = float(ROBUST_HYPERPARAMETERS["bgr"]["kappa_y"])
BGR_MAX_IRLS = 12
BGR_TOL = 2e-5
SCHWEPPE_MAX_IRLS = 12
SCHWEPPE_TOL = 2e-5
POOL_SEED = 20260803
DEV_FRACTION = 0.20


@dataclass
class Fitted:
    model: Ridge
    geom_weights: np.ndarray
    resid_weights: np.ndarray
    combined_weights: np.ndarray
    n_iter: int


@dataclass
class SchweppeFit:
    model: Ridge
    leverage_factor: np.ndarray
    weights: np.ndarray
    n_iter: int


def fit_schweppe(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    kappa_x: float,
    kappa_y: float,
    scale: float,
    max_iter: int = SCHWEPPE_MAX_IRLS,
    tol: float = SCHWEPPE_TOL,
) -> SchweppeFit:
    """Schweppe-type GM fit used identically in calibration and evaluation."""
    norms = np.linalg.norm(X, axis=1)
    v = np.minimum(1.0, kappa_x / np.maximum(norms, 1e-12))
    model = make_ridge(alpha)
    model.fit(X, y)
    previous = np.asarray(model.coef_).copy()
    weights = np.ones(len(y))
    for iteration in range(1, max_iter + 1):
        u = (y - model.predict(X)) / max(scale, 1e-12)
        weights = np.minimum(1.0, kappa_y * v / np.maximum(np.abs(u), 1e-12))
        new = make_ridge(alpha)
        new.fit(X, y, sample_weight=weights)
        current = np.asarray(new.coef_).copy()
        relative = np.linalg.norm(current - previous) / (np.linalg.norm(previous) + 1e-12)
        model, previous = new, current
        if relative < tol:
            return SchweppeFit(model=model, leverage_factor=v, weights=weights, n_iter=iteration)
    return SchweppeFit(model=model, leverage_factor=v, weights=weights, n_iter=max_iter)


def scale_for_geometry_budget(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    alpha: float,
    kappa_x: float,
) -> float:
    """Robust residual scale used consistently in calibration and evaluation."""
    wx, _ = geometric_weights(X, kappa_x)
    scale_model = make_ridge(alpha)
    scale_model.fit(X, y, sample_weight=wx)
    return robust_scale_from_unique_groups(scale_model, X, y, groups)


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", RobustScaler(quantile_range=(25, 75)), NUM_COLS),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=5,
                    sparse_output=False,
                ),
                CAT_COLS,
            ),
        ],
        sparse_threshold=0,
    )


def add_penalized_intercept(X: np.ndarray) -> np.ndarray:
    """Append a constant coordinate that is penalized by Ridge.

    Using ``fit_intercept=False`` aligns the implementation with the strongly
    convex objective in the stability theorem, including the constant direction.
    """
    X = np.asarray(X, dtype=float)
    return np.column_stack([np.ones(len(X), dtype=float), X])


def make_ridge(alpha: float) -> Ridge:
    return Ridge(alpha=float(alpha), fit_intercept=False, solver="cholesky", tol=1e-9, max_iter=3000)


def development_evaluation_pools(
    df: pd.DataFrame,
    pool_seed: int = POOL_SEED,
    dev_fraction: float = DEV_FRACTION,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Reserve globally disjoint listing-ID pools for development and evaluation."""
    ids = np.sort(df["idannonce"].drop_duplicates().to_numpy())
    rng = np.random.default_rng(pool_seed)
    shuffled = rng.permutation(ids)
    n_dev = max(1, min(len(ids) - 1, int(round(dev_fraction * len(ids)))))
    dev_ids = np.sort(shuffled[:n_dev])
    eval_ids = np.sort(shuffled[n_dev:])
    dev_set = set(dev_ids.tolist())
    eval_set = set(eval_ids.tolist())
    if dev_set & eval_set:
        raise RuntimeError("Development and evaluation ID pools overlap")

    import hashlib
    def digest(values: np.ndarray) -> str:
        payload = "\n".join(map(str, values.tolist())).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    metadata = {
        "pool_seed": int(pool_seed),
        "dev_fraction": float(dev_fraction),
        "total_unique_ids": int(len(ids)),
        "development_unique_ids": int(len(dev_ids)),
        "evaluation_unique_ids": int(len(eval_ids)),
        "intersection_size": 0,
        "development_ids_sha256": digest(dev_ids),
        "evaluation_ids_sha256": digest(eval_ids),
    }
    dev = df[df["idannonce"].isin(dev_set)].reset_index(drop=True).copy()
    evaluation = df[df["idannonce"].isin(eval_set)].reset_index(drop=True).copy()
    return dev, evaluation, metadata


def deterministic_listing_representatives(frame: pd.DataFrame) -> pd.DataFrame:
    """Return one order-invariant representative per listing.

    Numeric columns are aggregated by their median. Non-numeric columns use the
    lexicographically first modal value (with missing values retained as a
    possible category). Exact copies are unchanged. This convention matters
    only for the small number of IDs whose raw rows are not identical.
    """
    if frame.empty:
        return frame.copy()
    numeric = set(frame.select_dtypes(include=[np.number]).columns)
    rows = []
    for listing_id, group in frame.groupby("idannonce", sort=True, dropna=False):
        row = {}
        for col in frame.columns:
            values = group[col]
            if col == "idannonce":
                row[col] = listing_id
            elif col in numeric:
                row[col] = float(np.nanmedian(values.to_numpy(dtype=float)))
            else:
                counts = values.astype(object).where(values.notna(), "<NA>").value_counts(dropna=False)
                max_count = int(counts.max())
                winners = sorted([str(v) for v, c in counts.items() if int(c) == max_count])
                chosen = winners[0]
                row[col] = np.nan if chosen == "<NA>" else chosen
        rows.append(row)
    out = pd.DataFrame(rows, columns=frame.columns)
    return out.reset_index(drop=True)


def split_data(df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    groups = df["idannonce"].to_numpy()
    g1 = GroupShuffleSplit(n_splits=1, train_size=0.60, random_state=seed)
    train_idx, temp_idx = next(g1.split(df, groups=groups))
    temp = df.iloc[temp_idx]
    g2 = GroupShuffleSplit(n_splits=1, train_size=0.50, random_state=seed + 10000)
    val_rel, test_rel = next(g2.split(temp, groups=temp["idannonce"].to_numpy()))
    val_idx = temp_idx[val_rel]
    test_idx = temp_idx[test_rel]

    # Training remains raw, including repeated rows. Validation and test use an
    # order-invariant deterministic representative per listing.
    train = df.iloc[train_idx].reset_index(drop=True).copy()
    val = deterministic_listing_representatives(df.iloc[val_idx])
    test = deterministic_listing_representatives(df.iloc[test_idx])
    return train, val, test


def tune_alpha(X: np.ndarray, y: np.ndarray, Xv: np.ndarray, yv: np.ndarray) -> float:
    best_alpha = None
    best_rmse = math.inf
    for alpha in ALPHAS:
        model = make_ridge(float(alpha))
        model.fit(X, y)
        rmse = float(np.sqrt(mean_squared_error(yv, model.predict(Xv))))
        if rmse < best_rmse:
            best_rmse = rmse
            best_alpha = float(alpha)
    assert best_alpha is not None
    return best_alpha


def geometric_weights(X: np.ndarray, kappa_x: float) -> tuple[np.ndarray, np.ndarray]:
    norms = np.linalg.norm(X, axis=1)
    if not np.isfinite(kappa_x):
        return np.ones(len(X)), norms
    weights = np.ones(len(X))
    mask = norms > kappa_x
    weights[mask] = (kappa_x / np.maximum(norms[mask], 1e-12)) ** 2
    return weights, norms


def robust_scale_from_unique_groups(
    model: Ridge,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
) -> float:
    _, first = np.unique(groups, return_index=True)
    residual = y[first] - model.predict(X[first])
    center = np.median(residual)
    scale = 1.4826 * np.median(np.abs(residual - center))
    return float(max(scale, 1e-4))


def fit_method(
    method: str,
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    kappa_x: float,
    kappa_y: float,
    scale: float,
    tol: float = BGR_TOL,
    max_iter: int = BGR_MAX_IRLS,
) -> Fitted:
    use_geom = method in {"geometry", "bgr"}
    use_huber = method in {"huber", "bgr"}
    wx, _ = geometric_weights(X, kappa_x if use_geom else math.inf)

    model = make_ridge(alpha)
    model.fit(X, y, sample_weight=wx)

    if not use_huber:
        ones = np.ones(len(y))
        return Fitted(model, wx, ones, wx, 1)

    coef_prev = np.asarray(model.coef_).copy()
    wy = np.ones(len(y))
    n_iter = 1
    for n_iter in range(1, max_iter + 1):
        residual = y - model.predict(X)
        z = np.abs(residual) / max(scale, 1e-12)
        wy = np.ones(len(y))
        mask = z > kappa_y
        wy[mask] = kappa_y / np.maximum(z[mask], 1e-12)
        w = wx * wy
        new = make_ridge(alpha)
        new.fit(X, y, sample_weight=w)
        coef_change = np.linalg.norm(new.coef_ - coef_prev) / (np.linalg.norm(coef_prev) + 1e-12)
        model = new
        coef_prev = np.asarray(model.coef_).copy()
        if coef_change < tol:
            break
    return Fitted(model, wx, wy, wx * wy, n_iter)


def prediction_metrics(y_true: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    yt = y_true[mask]
    yp = pred[mask]
    err = yp - yt
    return {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "r2": float(r2_score(yt, yp)),
        "median_relative_error": float(np.median(np.abs(np.exp(np.clip(err, -10, 10)) - 1))),
    }


def select_singletons(train: pd.DataFrame, max_fraction: float, rng: np.random.Generator) -> np.ndarray:
    counts = train.groupby("idannonce").size()
    ids = counts[counts == 1].index.to_numpy()
    rng.shuffle(ids)
    n = max(1, int(round(max_fraction * train["idannonce"].nunique())))
    return ids[: min(n, len(ids))]


def row_indices_for_ids(train: pd.DataFrame, ids: np.ndarray) -> np.ndarray:
    return np.flatnonzero(train["idannonce"].isin(ids).to_numpy())


def group_noise_vector(train: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    ids = train["idannonce"].drop_duplicates().to_numpy()
    noise = rng.normal(size=len(ids))
    mapping = dict(zip(ids.tolist(), noise.tolist()))
    return train["idannonce"].map(mapping).to_numpy(dtype=float)


def realistic_cell_corruption(
    train: pd.DataFrame,
    ids: np.ndarray,
    modes: np.ndarray,
) -> pd.DataFrame:
    corrupted = train.copy()
    for listing_id, mode in zip(ids, modes):
        mask = corrupted["idannonce"].eq(listing_id)
        if mode == 0:
            corrupted.loc[mask, "surface"] = 0.0
        elif mode == 1:
            corrupted.loc[mask, "nb_pieces"] = 53
        else:
            corrupted.loc[mask, "nb_chambres"] = 22
    return corrupted


def append_fit_results(
    records: list[dict],
    seed: int,
    scenario: str,
    level: float,
    epsilon: float,
    method: str,
    fitted: Fitted,
    nominal_fitted: Fitted,
    X_test: np.ndarray,
    y_test: np.ndarray,
    plausible_mask: np.ndarray,
    corrupt_idx: np.ndarray | None,
    alpha: float,
    kappa_x: float,
    scale: float,
    label_amplitude: float | None = None,
) -> None:
    pred = fitted.model.predict(X_test)
    pred0 = nominal_fitted.model.predict(X_test)
    all_mask = np.ones(len(y_test), dtype=bool)
    all_m = prediction_metrics(y_test, pred, all_mask)
    clean_m = prediction_metrics(y_test, pred, plausible_mask)
    pred_diff = pred[plausible_mask] - pred0[plausible_mask]
    coef0 = np.asarray(nominal_fitted.model.coef_)
    coef = np.asarray(fitted.model.coef_)

    if corrupt_idx is None or len(corrupt_idx) == 0:
        gx = gy = gw = math.nan
    else:
        gx = float(np.mean(fitted.geom_weights[corrupt_idx]))
        gy = float(np.mean(fitted.resid_weights[corrupt_idx]))
        gw = float(np.mean(fitted.combined_weights[corrupt_idx]))

    records.append(
        {
            "seed": seed,
            "scenario": scenario,
            "level": float(level),
            "label_amplitude": math.nan if label_amplitude is None else float(label_amplitude),
            "epsilon": float(epsilon),
            "method": method,
            "alpha": alpha,
            "kappa_x": kappa_x,
            "kappa_y": KAPPA_Y,
            "scale": scale,
            "rmse_all": all_m["rmse"],
            "rmse_plausible": clean_m["rmse"],
            "mae_plausible": clean_m["mae"],
            "r2_plausible": clean_m["r2"],
            "median_relative_error_plausible": clean_m["median_relative_error"],
            "prediction_drift_rmse": float(np.sqrt(np.mean(pred_diff**2))),
            "prediction_drift_p95": float(np.quantile(np.abs(pred_diff), 0.95)),
            "coefficient_drift_l2": float(np.linalg.norm(coef - coef0)),
            "mean_geom_weight": float(np.mean(fitted.geom_weights)),
            "fraction_geom_clipped": float(np.mean(fitted.geom_weights < 0.999)),
            "mean_resid_weight": float(np.mean(fitted.resid_weights)),
            "fraction_resid_clipped": float(np.mean(fitted.resid_weights < 0.999)),
            "mean_corrupt_geom_weight": gx,
            "mean_corrupt_resid_weight": gy,
            "mean_corrupt_combined_weight": gw,
            "irls_iterations": fitted.n_iter,
        }
    )


def main() -> None:
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)
    df_all = pd.read_csv(INPUT)
    for col in CAT_COLS:
        df_all[col] = df_all[col].astype(str)
    dev_df, df, pool_metadata = development_evaluation_pools(df_all)
    (OUTDIR / "seloger_dev_eval_partition.json").write_text(
        json.dumps(pool_metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if (df["prix"] <= 0).any():
        raise ValueError("Non-positive prices cannot be log-transformed")

    records: list[dict] = []
    calibration: list[dict] = []

    for split_no, seed in enumerate(EVAL_SEEDS, start=1):
        print(f"split {split_no}/{len(EVAL_SEEDS)} seed={seed}", flush=True)
        train, val, test = split_data(df, seed)
        pre = make_preprocessor()
        X_train = add_penalized_intercept(np.asarray(pre.fit_transform(train[FEATURE_COLS]), dtype=float))
        X_val = add_penalized_intercept(np.asarray(pre.transform(val[FEATURE_COLS]), dtype=float))
        X_test = add_penalized_intercept(np.asarray(pre.transform(test[FEATURE_COLS]), dtype=float))
        y_train = np.log(train["prix"].to_numpy(dtype=float))
        y_val = np.log(val["prix"].to_numpy(dtype=float))
        y_test = np.log(test["prix"].to_numpy(dtype=float))
        groups = train["idannonce"].to_numpy()

        alpha = tune_alpha(X_train, y_train, X_val, y_val)
        norms = np.linalg.norm(X_train, axis=1)
        kappa_x = float(np.quantile(norms, Q_GEOMETRY))

        scale = scale_for_geometry_budget(
            X_train, y_train, groups, alpha, kappa_x
        )

        plausible_mask = (
            (test["surface"].to_numpy(dtype=float) > 0)
            & (test["nb_pieces"].to_numpy(dtype=float) <= 20)
            & (test["nb_chambres"].to_numpy(dtype=float) <= 10)
        )

        calibration.append(
            {
                "seed": seed,
                "alpha": alpha,
                "kappa_x": kappa_x,
                "scale": scale,
                "n_train_rows": len(train),
                "n_train_ids": train["idannonce"].nunique(),
                "n_test_ids": len(test),
                "n_test_plausible": int(plausible_mask.sum()),
            }
        )

        nominal: dict[str, Fitted] = {}
        for method in METHODS:
            nominal[method] = fit_method(method, X_train, y_train, alpha, kappa_x, KAPPA_Y, scale)
            append_fit_results(
                records,
                seed,
                "nominal_raw",
                0.0,
                0.0,
                method,
                nominal[method],
                nominal[method],
                X_test,
                y_test,
                plausible_mask,
                None,
                alpha,
                kappa_x,
                scale,
            )

        # Fixed singleton contamination pool: duplicates remain in the raw background,
        # but stress points are single listings so that the experiment is not a
        # duplication test in disguise.
        rng = np.random.default_rng(seed + 50000)
        master_ids = select_singletons(train, 0.20, rng)
        n_master = len(master_ids)
        x_cols = rng.integers(1, 1 + len(NUM_COLS), size=n_master)
        x_signs = rng.choice([-1.0, 1.0], size=n_master)
        y_signs = rng.choice([-1.0, 1.0], size=n_master)

        # Core stress scenarios at 5% contamination. The broader amplitude and
        # fraction sweeps from earlier exploratory versions are intentionally
        # omitted here so that the final protocol can be rerun end-to-end with
        # globally held-out development and evaluation pools.
        n5 = max(1, int(round(0.05 * train["idannonce"].nunique())))
        ids5 = master_ids[:n5]
        idx5 = row_indices_for_ids(train, ids5)
        cols5 = x_cols[: len(idx5)]
        xs5 = x_signs[: len(idx5)]
        ys5 = y_signs[: len(idx5)]

        # Gross vertical contamination.
        yc = y_train.copy()
        yc[idx5] += 4.0 * ys5
        for method in METHODS:
            fitted = fit_method(method, X_train, yc, alpha, kappa_x, KAPPA_Y, scale)
            append_fit_results(records, seed, "vertical_amplitude", 4.0, 0.05, method,
                               fitted, nominal[method], X_test, y_test, plausible_mask, idx5,
                               alpha, kappa_x, scale)

        # Gross leverage contamination.
        Xc = X_train.copy()
        Xc[idx5, cols5] += 50.0 * xs5
        for method in METHODS:
            fitted = fit_method(method, Xc, y_train, alpha, kappa_x, KAPPA_Y, scale)
            append_fit_results(records, seed, "leverage_amplitude", 50.0, 0.05, method,
                               fitted, nominal[method], X_test, y_test, plausible_mask, idx5,
                               alpha, kappa_x, scale)

        # Joint contamination of covariates and labels.
        Xm = X_train.copy()
        ym = y_train.copy()
        Xm[idx5, cols5] += 50.0 * xs5
        ym[idx5] += 4.0 * ys5
        for method in METHODS:
            fitted = fit_method(method, Xm, ym, alpha, kappa_x, KAPPA_Y, scale)
            append_fit_results(records, seed, "mixed_amplitude", 50.0, 0.05, method,
                               fitted, nominal[method], X_test, y_test, plausible_mask, idx5,
                               alpha, kappa_x, scale, label_amplitude=4.0)

        # Diffuse Gaussian label noise shared by all copies of a listing.
        noise_base = group_noise_vector(train, np.random.default_rng(seed + 60000))
        yc = y_train + 0.40 * noise_base
        all_idx = np.arange(len(train))
        for method in METHODS:
            fitted = fit_method(method, X_train, yc, alpha, kappa_x, KAPPA_Y, scale)
            append_fit_results(records, seed, "gaussian_noise", 0.40, 1.0, method,
                               fitted, nominal[method], X_test, y_test, plausible_mask, all_idx,
                               alpha, kappa_x, scale)

        # SeLoger-like cell corruptions: surface=0, pieces=53, bedrooms=22.
        realistic_ids = master_ids[:n5]
        realistic_modes = np.random.default_rng(seed + 70000).integers(0, 3, size=len(realistic_ids))
        train_bad = realistic_cell_corruption(train, realistic_ids, realistic_modes)
        X_bad = add_penalized_intercept(np.asarray(pre.transform(train_bad[FEATURE_COLS]), dtype=float))
        idx_bad = row_indices_for_ids(train, realistic_ids)
        for method in METHODS:
            fitted = fit_method(method, X_bad, y_train, alpha, kappa_x, KAPPA_Y, scale)
            append_fit_results(records, seed, "realistic_cells", 1.0, 0.05, method,
                               fitted, nominal[method], X_test, y_test, plausible_mask, idx_bad,
                               alpha, kappa_x, scale)

    detail = pd.DataFrame.from_records(records)

    # Add method-specific degradation relative to the nominal fit of the same method.
    nominal_rmse_plausible = (
        detail[detail["scenario"] == "nominal_raw"]
        .set_index(["seed", "method"])["rmse_plausible"]
        .to_dict()
    )
    nominal_rmse_all = (
        detail[detail["scenario"] == "nominal_raw"]
        .set_index(["seed", "method"])["rmse_all"]
        .to_dict()
    )
    detail["delta_rmse_plausible"] = [
        row.rmse_plausible - nominal_rmse_plausible[(row.seed, row.method)] for row in detail.itertuples()
    ]
    detail["delta_rmse_all"] = [
        row.rmse_all - nominal_rmse_all[(row.seed, row.method)] for row in detail.itertuples()
    ]
    detail.to_csv(DETAIL_PATH, index=False)
    pd.DataFrame(calibration).to_csv(OUTDIR / "seloger_bgr_calibration.csv", index=False)

    summary = (
        detail.groupby(["scenario", "level", "epsilon", "method"], as_index=False)
        .agg(
            n=("seed", "size"),
            rmse_all_mean=("rmse_all", "mean"),
            rmse_all_std=("rmse_all", "std"),
            rmse_plausible_mean=("rmse_plausible", "mean"),
            rmse_plausible_std=("rmse_plausible", "std"),
            delta_rmse_all_mean=("delta_rmse_all", "mean"),
            delta_rmse_all_std=("delta_rmse_all", "std"),
            delta_rmse_mean=("delta_rmse_plausible", "mean"),
            delta_rmse_std=("delta_rmse_plausible", "std"),
            prediction_drift_mean=("prediction_drift_rmse", "mean"),
            prediction_drift_std=("prediction_drift_rmse", "std"),
            p95_drift_mean=("prediction_drift_p95", "mean"),
            median_relative_error_mean=("median_relative_error_plausible", "mean"),
            mean_corrupt_geom_weight=("mean_corrupt_geom_weight", "mean"),
            mean_corrupt_resid_weight=("mean_corrupt_resid_weight", "mean"),
            mean_corrupt_combined_weight=("mean_corrupt_combined_weight", "mean"),
        )
    )
    summary["delta_rmse_all_ci95"] = 1.96 * summary["delta_rmse_all_std"] / np.sqrt(summary["n"])
    summary["delta_rmse_ci95"] = 1.96 * summary["delta_rmse_std"] / np.sqrt(summary["n"])
    summary["prediction_drift_ci95"] = 1.96 * summary["prediction_drift_std"] / np.sqrt(summary["n"])
    summary.to_csv(SUMMARY_PATH, index=False)

    # BGR vs Ridge paired comparisons.
    pair_rows = []
    keys = ["scenario", "level", "epsilon"]
    for key, group in detail.groupby(keys):
        pivot = group.pivot(index="seed", columns="method", values="rmse_plausible")
        if not {"ridge", "bgr"}.issubset(pivot.columns):
            continue
        diff = pivot["bgr"] - pivot["ridge"]
        if len(diff) > 1:
            ci = float(1.96 * diff.std(ddof=1) / math.sqrt(len(diff)))
        else:
            ci = math.nan
        pair_rows.append(
            {
                "scenario": key[0],
                "level": key[1],
                "epsilon": key[2],
                "n": len(diff),
                "mean_bgr_minus_ridge": float(diff.mean()),
                "ci95_half_width": ci,
                "bgr_wins": int((diff < 0).sum()),
                "min_bgr_minus_ridge": float(diff.min()),
                "max_bgr_minus_ridge": float(diff.max()),
                "note": "descriptive repeated-split summary; overlapping splits are not treated as independent replicates",
            }
        )
    pair = pd.DataFrame(pair_rows)
    pair.to_csv(PAIR_PATH, index=False)

    # Compact grouped figure for the final held-out protocol.
    core_scenarios = [
        ("vertical_amplitude", 4.0, "Labels"),
        ("leverage_amplitude", 50.0, "Leverage"),
        ("mixed_amplitude", 50.0, "Mixte"),
        ("gaussian_noise", 0.4, "Gaussien"),
        ("realistic_cells", 1.0, "Cellules"),
    ]
    bar_rows = []
    for scenario, level, label in core_scenarios:
        subset = summary[(summary.scenario == scenario) & np.isclose(summary.level, level)]
        for method in METHODS:
            row = subset[subset.method == method]
            if not row.empty:
                bar_rows.append({"scenario_label": label, "method": method,
                                 "delta_rmse": float(row.iloc[0].delta_rmse_mean)})
    bar = pd.DataFrame(bar_rows)
    x = np.arange(len(core_scenarios), dtype=float)
    width = 0.19
    plt.figure(figsize=(8.8, 5.0))
    for j, method in enumerate(METHODS):
        values = [float(bar[(bar.scenario_label == label) & (bar.method == method)].delta_rmse.iloc[0])
                  for _, _, label in core_scenarios]
        plt.bar(x + (j - 1.5) * width, values, width=width, label=method)
    plt.xticks(x, [label for _, _, label in core_scenarios])
    plt.ylabel("Augmentation de RMSE plausible")
    plt.title("SeLoger : dégradation sous contamination (protocole tenu à l'écart)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / "seloger_bgr_core_scenarios.png", dpi=180)
    plt.savefig(OUTDIR / "seloger_bgr_core_scenarios.pdf")
    plt.close()

    figure_specs = [(None, None, None, "seloger_bgr_core_scenarios.png")]

    # Compact JSON with the load-bearing comparisons.
    def summary_row(scenario: str, level: float, method: str) -> dict:
        row = summary[(summary.scenario == scenario) & np.isclose(summary.level, level) & (summary.method == method)].iloc[0]
        return {k: (float(row[k]) if np.isscalar(row[k]) else row[k]) for k in [
            "rmse_plausible_mean", "delta_rmse_mean", "delta_rmse_ci95",
            "prediction_drift_mean", "p95_drift_mean",
            "mean_corrupt_geom_weight", "mean_corrupt_resid_weight",
            "mean_corrupt_combined_weight"
        ]}

    nominal_table = summary[summary.scenario == "nominal_raw"][["method", "rmse_plausible_mean", "rmse_plausible_std", "median_relative_error_mean"]]
    compact = {
        "dataset": {
            "rows_total_raw": int(len(df_all)),
            "unique_listings_total": int(df_all.idannonce.nunique()),
            "evaluation_pool_rows": int(len(df)),
            "evaluation_pool_unique_listings": int(df.idannonce.nunique()),
            "development_pool_unique_listings": int(dev_df.idannonce.nunique()),
            "exact_duplicate_rows_beyond_first_total": int(df_all.duplicated().sum()),
        },
        "protocol": {
            "evaluation_splits": EVAL_SEEDS,
            "globally_disjoint_dev_eval_pools": pool_metadata,
            "training": "all raw rows",
            "evaluation": "one deterministic representative per idannonce (numeric median, lexicographic modal category); plausible subset excludes only surface<=0, pieces>20, bedrooms>10",
            "q_geometry": Q_GEOMETRY,
            "kappa_y": KAPPA_Y,
            "hyperparameter_source": str(HYPERPARAM_PATH.name),
            "penalized_intercept": True,
            "target": "log(price)",
        },
        "nominal": nominal_table.to_dict(orient="records"),
        "vertical_A4": {m: summary_row("vertical_amplitude", 4.0, m) for m in METHODS},
        "leverage_A50": {m: summary_row("leverage_amplitude", 50.0, m) for m in METHODS},
        "mixed_A50_y4": {m: summary_row("mixed_amplitude", 50.0, m) for m in METHODS},
        "gaussian_sigma_0_4": {m: summary_row("gaussian_noise", 0.4, m) for m in METHODS},
        "realistic_cells": {m: summary_row("realistic_cells", 1.0, m) for m in METHODS},
    }
    JSON_PATH.write_text(json.dumps(compact, indent=2, ensure_ascii=False), encoding="utf-8")

    # Package all artifacts.
    package_dir = OUTDIR / "seloger_bgr_experiment"
    if package_dir.exists():
        shutil.rmtree(package_dir)
    package_dir.mkdir()
    artifacts = [
        Path(__file__), DETAIL_PATH, SUMMARY_PATH, PAIR_PATH, JSON_PATH,
        OUTDIR / "seloger_bgr_calibration.csv",
        OUTDIR / "seloger_robust_hyperparameter_development.csv",
        OUTDIR / "seloger_dev_eval_partition.json",
        HYPERPARAM_PATH,
    ] + [OUTDIR / spec[3] for spec in figure_specs]
    for artifact in artifacts:
        if artifact.exists():
            shutil.copy2(artifact, package_dir / artifact.name)
    shutil.make_archive(str(OUTDIR / "seloger_bgr_experiment"), "zip", package_dir)

    print(f"wrote {DETAIL_PATH}")
    print(f"wrote {SUMMARY_PATH}")
    print(f"wrote {JSON_PATH}")


if __name__ == "__main__":
    main()
