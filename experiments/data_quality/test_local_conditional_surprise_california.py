#!/usr/bin/env python3
"""Surprise conditionnelle locale sur California Housing.

Objectif
--------
Tester si une lecture locale de l'innovation distingue mieux :
- un label artificiellement corrompu ;
- un point rare mais propre ;
- un point redondant/dupliqué ;
- un point ordinaire.

Le modèle de base reste une Ridge linéaire. Le score global est le résidu
studentisé leave-one-out. Le score local retire la médiane des résidus
studentisés de voisins géométriques et normalise par leur MAD locale.

Deux géométries de voisinage sont comparées :
1. distance euclidienne dans les variables standardisées ;
2. distance induite par la cométrique Ridge P :
       d_P(x_i,x_j)^2 = (x_i-x_j)^T P_x (x_i-x_j).

Les hyperparamètres de voisinage sont choisis uniquement sur la validation
propre, en minimisant la MSE d'une correction locale du biais résiduel. Le jeu
de test n'intervient jamais dans le calcul des scores ou des poids.
"""

from __future__ import annotations

import argparse
import json
import math
import zipfile
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.datasets import fetch_california_housing
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from scipy.spatial import cKDTree
from sklearn.preprocessing import StandardScaler


FEATURE_NAMES = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]


def load_california(data_path: Path, archive_path: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    if not data_path.exists():
        if archive_path is None or not archive_path.exists():
            raise FileNotFoundError(f"Données absentes : {data_path}")
        with zipfile.ZipFile(archive_path) as archive:
            archive.extract("california_housing_raw.csv", data_path.parent)

    frame = pd.read_csv(data_path)
    households = frame["households"].to_numpy(dtype=float)
    x = np.column_stack(
        [
            frame["median_income"].to_numpy(dtype=float),
            frame["housing_median_age"].to_numpy(dtype=float),
            frame["total_rooms"].to_numpy(dtype=float) / households,
            frame["total_bedrooms"].to_numpy(dtype=float) / households,
            frame["population"].to_numpy(dtype=float),
            frame["population"].to_numpy(dtype=float) / households,
            frame["latitude"].to_numpy(dtype=float),
            frame["longitude"].to_numpy(dtype=float),
        ]
    )
    y = frame["median_house_value"].to_numpy(dtype=float)
    return x, y


def penalty_matrix(p: int) -> np.ndarray:
    m = np.eye(p, dtype=float)
    m[0, 0] = 0.0
    return m


def ridge_fit(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    p = x.shape[1]
    m = penalty_matrix(p)
    if weights is None:
        precision = x.T @ x + alpha * m
        rhs = x.T @ y
    else:
        w = np.asarray(weights, dtype=float)
        precision = x.T @ (w[:, None] * x) + alpha * m
        rhs = x.T @ (w * y)
    # Petite sécurité numérique, sans pénaliser l'intercept de façon substantielle.
    precision = precision + 1e-12 * np.eye(p)
    theta = np.linalg.solve(precision, rhs)
    covariance = np.linalg.inv(precision)
    return theta, covariance


def tune_alpha(
    x: np.ndarray,
    y: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    alphas: np.ndarray,
    weights: np.ndarray | None = None,
) -> tuple[float, float, np.ndarray]:
    best: tuple[float, float, np.ndarray] | None = None
    for alpha in alphas:
        theta, _ = ridge_fit(x, y, float(alpha), weights)
        mse = float(np.mean((x_val @ theta - y_val) ** 2))
        if best is None or mse < best[0]:
            best = (mse, float(alpha), theta)
    if best is None:
        raise RuntimeError("Grille d'alpha vide.")
    return best


def robust_scale(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    return max(1e-8, 1.4826 * mad)


def detection_metrics(labels: np.ndarray, score: np.ndarray, rare_clean: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=bool)
    score = np.asarray(score, dtype=float)
    k = max(1, int(np.sum(labels)))
    order = np.argsort(score)[::-1]
    top = order[:k]
    false_alarms = top[~labels[top]]
    return {
        "roc_auc": float(roc_auc_score(labels, score)),
        "average_precision": float(average_precision_score(labels, score)),
        "precision_at_k": float(np.mean(labels[top])),
        "rare_clean_fraction_in_top_k": float(np.mean(rare_clean[top])),
        "rare_clean_share_of_false_alarms": float(
            np.sum(rare_clean[false_alarms]) / max(len(false_alarms), 1)
        ),
    }


def matrix_sqrt_psd(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh((matrix + matrix.T) / 2.0)
    values = np.clip(values, 0.0, None)
    return vectors @ np.diag(np.sqrt(values))


def query_neighbors(
    train_embedding: np.ndarray,
    query_embedding: np.ndarray,
    max_k: int,
    self_query: bool,
) -> tuple[np.ndarray, np.ndarray]:
    extra = 1 if self_query else 0
    tree = cKDTree(np.asarray(train_embedding, dtype=float), compact_nodes=True, balanced_tree=True)
    distances, indices = tree.query(
        np.asarray(query_embedding, dtype=float),
        k=max_k + extra,
        workers=1,
    )
    distances = np.asarray(distances)
    indices = np.asarray(indices)
    if self_query:
        n = len(train_embedding)
        row_ids = np.arange(n)[:, None]
        mask = indices != row_ids
        indices = indices[mask].reshape(n, max_k)
        distances = distances[mask].reshape(n, max_k)
    return indices.astype(np.int64, copy=False), distances

def local_median_mad(
    reference_values: np.ndarray,
    neighbor_indices: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    values = reference_values[neighbor_indices[:, :k]]
    location = np.median(values, axis=1)
    mad = 1.4826 * np.median(np.abs(values - location[:, None]), axis=1)
    # Un plancher très faible évite les divisions par zéro dans les amas exacts.
    scale = np.maximum(mad, 0.05)
    return location, scale


def choose_local_model(
    z_train: np.ndarray,
    z_val_raw: np.ndarray,
    pred_val: np.ndarray,
    y_val: np.ndarray,
    train_indices: np.ndarray,
    train_distances: np.ndarray,
    val_indices: np.ndarray,
    val_distances: np.ndarray,
    k_grid: list[int],
) -> dict[str, Any]:
    """Choisit une correction locale avec repli explicite vers le score global.

    Dans les unités du score global, on écrit

        z = b(x) + epsilon.

    La médiane des voisins estime le biais local b(x). Sa contribution est
    rétractée par ``location_shrink``. La variance locale est elle aussi
    rétractée vers la variance globale 1 par ``scale_mix``. Le modèle global
    (location_shrink=0, scale_mix=0) appartient donc à la grille : la version
    locale ne peut être imposée quand la validation ne la justifie pas.
    """
    best: dict[str, Any] | None = None
    location_grid = [0.0, 0.25, 0.5, 0.75, 1.0]
    scale_mix_grid = [0.0, 0.25, 0.5, 0.75, 1.0]
    gamma_grid = [0.0, 0.25, 0.5, 1.0]

    for k in k_grid:
        val_location, val_scale_local = local_median_mad(z_train, val_indices, k)
        radius_ref = max(float(np.median(train_distances[:, k - 1])), 1e-12)
        radius_ratio_val = val_distances[:, k - 1] / radius_ref

        for location_shrink in location_grid:
            centered = z_val_raw - location_shrink * val_location
            for scale_mix in scale_mix_grid:
                # Rétraction de la variance locale vers la variance globale 1.
                noise_var = (1.0 - scale_mix) + scale_mix * val_scale_local**2
                # Incertitude de la médiane locale (approximation gaussienne).
                base_var = noise_var * (1.0 + math.pi / (2.0 * k))
                for gamma in gamma_grid:
                    # Une région plus isolée peut recevoir une variance
                    # supplémentaire, mais seulement au-delà du rayon médian.
                    density_var = gamma * np.maximum(radius_ratio_val**2 - 1.0, 0.0)
                    shape_scale = np.sqrt(np.maximum(base_var + density_var, 1e-10))
                    standardized = centered / shape_scale
                    calibration = max(
                        float(np.median(np.abs(standardized)) / 0.67448975),
                        1e-6,
                    )
                    final_scale = calibration * shape_scale
                    nll = float(
                        np.mean(
                            np.log(final_scale)
                            + 0.5 * (centered / final_scale) ** 2
                        )
                    )
                    candidate = {
                        "criterion": nll,
                        "nll": nll,
                        "k": int(k),
                        "location_shrink": float(location_shrink),
                        "scale_mix": float(scale_mix),
                        "gamma": float(gamma),
                        "calibration": calibration,
                        "radius_ref": radius_ref,
                    }
                    if best is None or candidate["criterion"] < best["criterion"]:
                        best = candidate

    if best is None:
        raise RuntimeError("Impossible de sélectionner le voisinage local.")
    return best

def local_scores_from_neighbors(
    z_train: np.ndarray,
    indices: np.ndarray,
    distances: np.ndarray,
    model: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    k = int(model["k"])
    location_shrink = float(model["location_shrink"])
    scale_mix = float(model["scale_mix"])
    gamma = float(model["gamma"])
    calibration = float(model["calibration"])
    radius_ref = float(model["radius_ref"])
    location, scale_local = local_median_mad(z_train, indices, k)
    radius_ratio = distances[:, k - 1] / max(radius_ref, 1e-12)
    noise_var = (1.0 - scale_mix) + scale_mix * scale_local**2
    base_var = noise_var * (1.0 + math.pi / (2.0 * k))
    density_var = gamma * np.maximum(radius_ratio**2 - 1.0, 0.0)
    scale = calibration * np.sqrt(np.maximum(base_var + density_var, 1e-10))
    score = (z_train - location_shrink * location) / np.maximum(scale, 1e-8)
    return score, location, scale, radius_ratio

def paired_statistics(differences: np.ndarray) -> dict[str, float]:
    differences = np.asarray(differences, dtype=float)
    n = len(differences)
    mean = float(np.mean(differences))
    sd = float(np.std(differences, ddof=1)) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 1 else math.nan
    crit = float(stats.t.ppf(0.975, n - 1)) if n > 1 else math.nan
    try:
        wilcoxon_p = float(stats.wilcoxon(differences).pvalue)
    except ValueError:
        wilcoxon_p = math.nan
    return {
        "mean_difference": mean,
        "sd_difference": sd,
        "ci95_low": mean - crit * se if n > 1 else math.nan,
        "ci95_high": mean + crit * se if n > 1 else math.nan,
        "paired_t_pvalue": float(stats.ttest_1samp(differences, 0.0).pvalue) if n > 1 else math.nan,
        "wilcoxon_pvalue": wilcoxon_p,
        "wins_negative_difference": int(np.sum(differences < 0.0)),
        "losses_positive_difference": int(np.sum(differences > 0.0)),
    }


def run_one_split(
    x_all: np.ndarray,
    y_all: np.ndarray,
    seed: int,
    corruption_magnitude: float,
    corruption_fraction: float = 0.05,
    duplicate_fraction: float = 0.05,
    duplicate_copies: int = 3,
    k_grid: list[int] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    if k_grid is None:
        k_grid = [16, 32, 64, 128, 256]
    max_k = max(k_grid)
    rng = np.random.default_rng(50_000 + seed + int(1000 * corruption_magnitude))

    x_train_raw, x_temp_raw, y_train_raw, y_temp_raw = train_test_split(
        x_all, y_all, test_size=0.40, random_state=seed
    )
    x_val_raw, x_test_raw, y_val_raw, y_test_raw = train_test_split(
        x_temp_raw, y_temp_raw, test_size=0.50, random_state=1_000 + seed
    )

    imputer = SimpleImputer(strategy="median").fit(x_train_raw)
    x_train_imp = imputer.transform(x_train_raw)
    x_val_imp = imputer.transform(x_val_raw)
    x_test_imp = imputer.transform(x_test_raw)

    x_scaler = StandardScaler().fit(x_train_imp)
    x_train = x_scaler.transform(x_train_imp)
    x_val = x_scaler.transform(x_val_imp)
    x_test = x_scaler.transform(x_test_imp)

    y_scaler = StandardScaler().fit(y_train_raw.reshape(-1, 1))
    y_train_clean = y_scaler.transform(y_train_raw.reshape(-1, 1)).ravel()
    y_val = y_scaler.transform(y_val_raw.reshape(-1, 1)).ravel()
    y_test = y_scaler.transform(y_test_raw.reshape(-1, 1)).ravel()

    feature_covariance = np.cov(x_train, rowvar=False)
    feature_precision = np.linalg.inv(feature_covariance + 1e-6 * np.eye(x_train.shape[1]))
    mahalanobis2 = np.einsum("ij,jk,ik->i", x_train, feature_precision, x_train)
    n_rare = max(1, int(0.05 * len(x_train)))
    rare_original = np.zeros(len(x_train), dtype=bool)
    rare_original[np.argsort(mahalanobis2)[-n_rare:]] = True

    central_candidates = np.where((~rare_original) & (mahalanobis2 < np.quantile(mahalanobis2, 0.75)))[0]
    n_duplicate_bases = max(1, int(duplicate_fraction * len(x_train)))
    duplicate_bases = rng.choice(central_candidates, size=n_duplicate_bases, replace=False)
    duplicate_base_mask = np.zeros(len(x_train), dtype=bool)
    duplicate_base_mask[duplicate_bases] = True

    eligible = np.where((~rare_original) & (~duplicate_base_mask))[0]
    n_corrupted = max(1, int(corruption_fraction * len(x_train)))
    corrupted_indices = rng.choice(eligible, size=n_corrupted, replace=False)
    corrupted_original = np.zeros(len(x_train), dtype=bool)
    corrupted_original[corrupted_indices] = True

    y_train_observed = y_train_clean.copy()
    signs = rng.choice(np.array([-1.0, 1.0]), size=n_corrupted)
    perturbation = signs * (corruption_magnitude + 0.25 * np.abs(rng.normal(size=n_corrupted)))
    y_train_observed[corrupted_indices] += perturbation

    x_copies = np.repeat(x_train[duplicate_bases], duplicate_copies, axis=0)
    y_copies = np.repeat(y_train_clean[duplicate_bases], duplicate_copies)
    x_aug = np.vstack([x_train, x_copies])
    y_aug = np.concatenate([y_train_observed, y_copies])

    n_added = len(x_copies)
    corrupted = np.concatenate([corrupted_original, np.zeros(n_added, dtype=bool)])
    rare_clean = np.concatenate([
        rare_original & (~corrupted_original) & (~duplicate_base_mask),
        np.zeros(n_added, dtype=bool),
    ])
    duplicate_group = np.concatenate([duplicate_base_mask, np.ones(n_added, dtype=bool)])
    ordinary_clean = (~corrupted) & (~rare_clean) & (~duplicate_group)

    x_aug_design = np.column_stack([np.ones(len(x_aug)), x_aug])
    x_train_clean_design = np.column_stack([np.ones(len(x_train)), x_train])
    x_val_design = np.column_stack([np.ones(len(x_val)), x_val])
    x_test_design = np.column_stack([np.ones(len(x_test)), x_test])

    alphas = np.r_[0.0, np.logspace(-4, 6, 51)]
    validation_mse, alpha, _ = tune_alpha(x_aug_design, y_aug, x_val_design, y_val, alphas)
    theta, covariance = ridge_fit(x_aug_design, y_aug, alpha)

    prediction_train = x_aug_design @ theta
    prediction_val = x_val_design @ theta
    residual = y_aug - prediction_train
    leverage = np.einsum("ij,jk,ik->i", x_aug_design, covariance, x_aug_design)
    leverage = np.clip(leverage, 0.0, 1.0 - 1e-10)
    sigma_global = robust_scale(residual / np.sqrt(np.maximum(1.0 - leverage, 1e-12)))
    z_global = residual / (sigma_global * np.sqrt(np.maximum(1.0 - leverage, 1e-12)))
    z_val_raw = (y_val - prediction_val) / sigma_global
    information = -0.5 * np.log1p(-leverage)

    # Deux géométries de voisinage.
    p_x = covariance[1:, 1:]
    p_sqrt = matrix_sqrt_psd(p_x)
    embeddings = {
        "euclidean": (x_aug, x_val),
        "ridge_geometry": (x_aug @ p_sqrt, x_val @ p_sqrt),
    }

    local_results: dict[str, dict[str, Any]] = {}
    local_scores: dict[str, np.ndarray] = {}
    local_radius: dict[str, np.ndarray] = {}

    for geometry_name, (train_embedding, val_embedding) in embeddings.items():
        train_idx, train_dist = query_neighbors(train_embedding, train_embedding, max_k, self_query=True)
        val_idx, val_dist = query_neighbors(train_embedding, val_embedding, max_k, self_query=False)
        model = choose_local_model(
            z_train=z_global,
            z_val_raw=z_val_raw,
            pred_val=prediction_val,
            y_val=y_val,
            train_indices=train_idx,
            train_distances=train_dist,
            val_indices=val_idx,
            val_distances=val_dist,
            k_grid=k_grid,
        )
        score, location, scale, radius_ratio = local_scores_from_neighbors(
            z_global, train_idx, train_dist, model
        )
        local_scores[geometry_name] = score
        local_radius[geometry_name] = radius_ratio
        local_results[geometry_name] = {
            "model": model,
            "detection": detection_metrics(corrupted, np.abs(score), rare_clean),
            "median_abs_score_ordinary": float(np.median(np.abs(score[ordinary_clean]))),
            "median_abs_score_rare": float(np.median(np.abs(score[rare_clean]))),
            "median_abs_score_corrupted": float(np.median(np.abs(score[corrupted]))),
            "median_radius_ordinary": float(np.median(radius_ratio[ordinary_clean])),
            "median_radius_rare": float(np.median(radius_ratio[rare_clean])),
        }

    detection_global = detection_metrics(corrupted, np.abs(z_global), rare_clean)

    # Pondération conservatrice : Huber c=3, une seule itération de repondération.
    weight_global = np.minimum(1.0, 3.0 / np.maximum(np.abs(z_global), 1e-12))
    z_local_geom = local_scores["ridge_geometry"]
    weight_local = np.minimum(1.0, 3.0 / np.maximum(np.abs(z_local_geom), 1e-12))

    _, alpha_w_global, theta_w_global = tune_alpha(
        x_aug_design, y_aug, x_val_design, y_val, alphas, weight_global
    )
    _, alpha_w_local, theta_w_local = tune_alpha(
        x_aug_design, y_aug, x_val_design, y_val, alphas, weight_local
    )
    oracle_weights = (~corrupted).astype(float)
    _, alpha_oracle, theta_oracle = tune_alpha(
        x_aug_design, y_aug, x_val_design, y_val, alphas, oracle_weights
    )
    _, alpha_clean, theta_clean = tune_alpha(
        x_train_clean_design, y_train_clean, x_val_design, y_val, alphas
    )

    methods = {
        "ridge_contamine": theta,
        "ridge_huber_global": theta_w_global,
        "ridge_huber_local_geometrique": theta_w_local,
        "ridge_oracle_exclusion": theta_oracle,
        "ridge_donnees_propres": theta_clean,
    }
    target_scale = float(y_scaler.scale_[0])
    test_results: dict[str, dict[str, float]] = {}
    for name, fitted in methods.items():
        pred = x_test_design @ fitted
        mse = float(np.mean((pred - y_test) ** 2))
        test_results[name] = {
            "test_mse_standardized": mse,
            "test_rmse_dollars": math.sqrt(mse) * target_scale,
        }

    def group_summary(mask: np.ndarray) -> dict[str, float]:
        return {
            "count": int(np.sum(mask)),
            "median_information": float(np.median(information[mask])),
            "median_abs_global_surprise": float(np.median(np.abs(z_global[mask]))),
            "median_abs_local_surprise": float(np.median(np.abs(z_local_geom[mask]))),
            "median_global_influence": float(np.median(leverage[mask] * z_global[mask] ** 2)),
            "median_local_influence": float(np.median(leverage[mask] * z_local_geom[mask] ** 2)),
            "mean_global_weight": float(np.mean(weight_global[mask])),
            "mean_local_weight": float(np.mean(weight_local[mask])),
            "median_local_radius_ratio": float(np.median(local_radius["ridge_geometry"][mask])),
        }

    groups = {
        "ordinary_clean": ordinary_clean,
        "rare_clean": rare_clean,
        "duplicate_group": duplicate_group,
        "corrupted": corrupted,
    }
    group_results = {name: group_summary(mask) for name, mask in groups.items()}

    summary: dict[str, Any] = {
        "seed": seed,
        "corruption_magnitude": corruption_magnitude,
        "alpha": alpha,
        "validation_mse": validation_mse,
        "sigma_global": sigma_global,
        "n_original_train": int(len(x_train)),
        "n_augmented_train": int(len(x_aug)),
        "n_corrupted": int(np.sum(corrupted)),
        "n_rare_clean": int(np.sum(rare_clean)),
        "n_duplicate_group": int(np.sum(duplicate_group)),
        "global_detection": detection_global,
        "local": local_results,
        "groups": group_results,
        "weights": {
            "alpha_global": alpha_w_global,
            "alpha_local": alpha_w_local,
            "alpha_oracle": alpha_oracle,
            "alpha_clean": alpha_clean,
        },
        "test": test_results,
    }

    point_frame = pd.DataFrame({
        "seed": seed,
        "corruption_magnitude": corruption_magnitude,
        "corrupted": corrupted,
        "rare_clean": rare_clean,
        "duplicate_group": duplicate_group,
        "ordinary_clean": ordinary_clean,
        "leverage": leverage,
        "information": information,
        "z_global": z_global,
        "z_local_euclidean": local_scores["euclidean"],
        "z_local_ridge_geometry": z_local_geom,
        "radius_ratio_ridge_geometry": local_radius["ridge_geometry"],
        "weight_global": weight_global,
        "weight_local": weight_local,
    })
    return summary, point_frame


def flatten_detection(results: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for result in results:
        base = {"seed": result["seed"], "corruption_magnitude": result["corruption_magnitude"]}
        row = dict(base)
        row.update({"score": "global"})
        row.update(result["global_detection"])
        rows.append(row)
        for geometry in ["euclidean", "ridge_geometry"]:
            row = dict(base)
            row.update({
                "score": f"local_{geometry}",
                "k": result["local"][geometry]["model"]["k"],
                "gamma": result["local"][geometry]["model"]["gamma"],
                "location_shrink": result["local"][geometry]["model"]["location_shrink"],
                "scale_mix": result["local"][geometry]["model"]["scale_mix"],
            })
            row.update(result["local"][geometry]["detection"])
            rows.append(row)
    return pd.DataFrame(rows)


def flatten_groups(results: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for result in results:
        for group, metrics in result["groups"].items():
            row = {
                "seed": result["seed"],
                "corruption_magnitude": result["corruption_magnitude"],
                "group": group,
            }
            row.update(metrics)
            rows.append(row)
    return pd.DataFrame(rows)


def flatten_test(results: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for result in results:
        for method, metrics in result["test"].items():
            row = {
                "seed": result["seed"],
                "corruption_magnitude": result["corruption_magnitude"],
                "method": method,
            }
            row.update(metrics)
            rows.append(row)
    return pd.DataFrame(rows)


def make_summary(
    detection_df: pd.DataFrame,
    group_df: pd.DataFrame,
    test_df: pd.DataFrame,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    output: dict[str, Any] = {"by_corruption_magnitude": {}}
    for magnitude in sorted(detection_df["corruption_magnitude"].unique()):
        key = str(magnitude)
        d = detection_df[detection_df["corruption_magnitude"] == magnitude]
        g = group_df[group_df["corruption_magnitude"] == magnitude]
        t = test_df[test_df["corruption_magnitude"] == magnitude]
        item: dict[str, Any] = {
            "detection_mean": d.groupby("score")[[
                "roc_auc", "average_precision", "precision_at_k",
                "rare_clean_fraction_in_top_k", "rare_clean_share_of_false_alarms"
            ]].mean().to_dict(orient="index"),
            "detection_sd": d.groupby("score")[["roc_auc", "average_precision", "precision_at_k"]].std().to_dict(orient="index"),
            "groups_mean": g.groupby("group")[[
                "median_information", "median_abs_global_surprise", "median_abs_local_surprise",
                "median_global_influence", "median_local_influence", "mean_global_weight",
                "mean_local_weight", "median_local_radius_ratio"
            ]].mean().to_dict(orient="index"),
            "test_mean": t.groupby("method")[["test_mse_standardized", "test_rmse_dollars"]].mean().to_dict(orient="index"),
            "test_sd": t.groupby("method")[["test_mse_standardized", "test_rmse_dollars"]].std().to_dict(orient="index"),
            "selected_k": {},
        }
        subset_results = [r for r in results if r["corruption_magnitude"] == magnitude]
        for geometry in ["euclidean", "ridge_geometry"]:
            ks = [r["local"][geometry]["model"]["k"] for r in subset_results]
            gammas = [r["local"][geometry]["model"]["gamma"] for r in subset_results]
            item["selected_k"][geometry] = {
                "mean_k": float(np.mean(ks)),
                "median_k": float(np.median(ks)),
                "k_counts": {str(k): int(np.sum(np.array(ks) == k)) for k in sorted(set(ks))},
                "gamma_mean": float(np.mean(gammas)),
                "location_shrink_mean": float(np.mean([r["local"][geometry]["model"]["location_shrink"] for r in subset_results])),
                "scale_mix_mean": float(np.mean([r["local"][geometry]["model"]["scale_mix"] for r in subset_results])),
            }

        # Comparaisons appariées de l'AUC locale géométrique et globale.
        pivot_auc = d.pivot(index="seed", columns="score", values="roc_auc")
        if "local_ridge_geometry" in pivot_auc and "global" in pivot_auc:
            item["paired_auc_local_minus_global"] = paired_statistics(
                pivot_auc["local_ridge_geometry"].to_numpy() - pivot_auc["global"].to_numpy()
            )
        if "local_ridge_geometry" in pivot_auc and "local_euclidean" in pivot_auc:
            item["paired_auc_geometry_minus_euclidean"] = paired_statistics(
                pivot_auc["local_ridge_geometry"].to_numpy() - pivot_auc["local_euclidean"].to_numpy()
            )

        pivot_mse = t.pivot(index="seed", columns="method", values="test_mse_standardized")
        if "ridge_huber_local_geometrique" in pivot_mse and "ridge_huber_global" in pivot_mse:
            item["paired_mse_local_weight_minus_global_weight"] = paired_statistics(
                pivot_mse["ridge_huber_local_geometrique"].to_numpy()
                - pivot_mse["ridge_huber_global"].to_numpy()
            )
        if "ridge_huber_local_geometrique" in pivot_mse and "ridge_contamine" in pivot_mse:
            item["paired_mse_local_weight_minus_baseline"] = paired_statistics(
                pivot_mse["ridge_huber_local_geometrique"].to_numpy()
                - pivot_mse["ridge_contamine"].to_numpy()
            )
        output["by_corruption_magnitude"][key] = item
    return output


def plot_detection(detection_df: pd.DataFrame, output_path: Path) -> None:
    grouped = detection_df.groupby(["corruption_magnitude", "score"])["roc_auc"].agg(["mean", "std"]).reset_index()
    magnitudes = sorted(grouped["corruption_magnitude"].unique())
    score_order = ["global", "local_euclidean", "local_ridge_geometry"]
    x = np.arange(len(magnitudes), dtype=float)
    width = 0.24
    fig, ax = plt.subplots(figsize=(9, 5.4))
    for offset, score in enumerate(score_order):
        part = grouped[grouped["score"] == score].set_index("corruption_magnitude").reindex(magnitudes)
        ax.bar(x + (offset - 1) * width, part["mean"], width=width, yerr=part["std"], capsize=3, label=score)
    ax.set_xticks(x, [str(m) for m in magnitudes])
    ax.set_ylim(0.45, 1.01)
    ax.set_xlabel("Amplitude de corruption (écarts-types de la cible)")
    ax.set_ylabel("AUC de détection")
    ax.set_title("Surprise globale et surprise conditionnelle locale")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_group_passport(group_df: pd.DataFrame, output_path: Path, magnitude: float) -> None:
    part = group_df[group_df["corruption_magnitude"] == magnitude]
    summary = part.groupby("group")[["median_information", "median_abs_local_surprise"]].mean()
    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    for group, row in summary.iterrows():
        ax.scatter(row["median_information"], row["median_abs_local_surprise"], s=100, label=group)
        ax.annotate(group, (row["median_information"], row["median_abs_local_surprise"]), xytext=(6, 5), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Information géométrique conditionnelle (médiane)")
    ax.set_ylabel("Surprise locale |z| (médiane)")
    ax.set_title(f"Passeport local des points — corruption {magnitude}σ")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_weights(group_df: pd.DataFrame, output_path: Path, magnitude: float) -> None:
    part = group_df[group_df["corruption_magnitude"] == magnitude]
    summary = part.groupby("group")[["mean_global_weight", "mean_local_weight"]].mean()
    groups = list(summary.index)
    x = np.arange(len(groups))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 5.3))
    ax.bar(x - width / 2, summary["mean_global_weight"], width=width, label="poids global")
    ax.bar(x + width / 2, summary["mean_local_weight"], width=width, label="poids local")
    ax.set_xticks(x, groups, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Poids Huber moyen")
    ax.set_title(f"Fiabilité globale vs locale — corruption {magnitude}σ")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("/mnt/data/california_housing_raw.csv"))
    parser.add_argument("--archive", type=Path, default=Path("/mnt/data/california_geometric_weighting_experiment.zip"))
    parser.add_argument("--output-dir", type=Path, default=Path("/mnt/data"))
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--magnitudes", type=float, nargs="+", default=[0.75, 1.5, 3.0])
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    x_all, y_all = load_california(args.data, args.archive)

    results: list[dict[str, Any]] = []
    point_frames: list[pd.DataFrame] = []
    for magnitude in args.magnitudes:
        for seed in range(args.seed_start, args.seed_start + args.seeds):
            print(f"magnitude={magnitude} seed={seed}", flush=True)
            result, points = run_one_split(x_all, y_all, seed, magnitude)
            results.append(result)
            # Conserver tous les points serait volumineux ; sous-échantillon stratifié
            # pour la figure et l'audit, mais les métriques utilisent tout le train.
            sample_parts = []
            for col in ["corrupted", "rare_clean", "duplicate_group", "ordinary_clean"]:
                mask = points[col].to_numpy(dtype=bool)
                part = points[mask]
                if len(part) > 1500:
                    part = part.sample(1500, random_state=seed)
                sample_parts.append(part)
            point_frames.append(pd.concat(sample_parts, ignore_index=True))
            import gc
            gc.collect()

    detection_df = flatten_detection(results)
    group_df = flatten_groups(results)
    test_df = flatten_test(results)
    point_df = pd.concat(point_frames, ignore_index=True)
    summary = make_summary(detection_df, group_df, test_df, results)

    detection_path = args.output_dir / "california_local_surprise_detection.csv"
    group_path = args.output_dir / "california_local_surprise_groups.csv"
    test_path = args.output_dir / "california_local_surprise_weighting.csv"
    point_path = args.output_dir / "california_local_surprise_points_sample.csv"
    summary_path = args.output_dir / "california_local_surprise_summary.json"
    detection_df.to_csv(detection_path, index=False)
    group_df.to_csv(group_path, index=False)
    test_df.to_csv(test_path, index=False)
    point_df.to_csv(point_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    plot_detection(detection_df, args.output_dir / "california_local_surprise_detection.png")
    plot_group_passport(group_df, args.output_dir / "california_local_surprise_passport.png", max(args.magnitudes))
    plot_weights(group_df, args.output_dir / "california_local_surprise_weights.png", max(args.magnitudes))

    readme = args.output_dir / "README_LOCAL_SURPRISE_CALIFORNIA.md"
    readme.write_text(
        "# Surprise conditionnelle locale — California Housing\n\n"
        "Exécution :\n\n"
        "```bash\npython test_local_conditional_surprise_california.py --seeds 20\n```\n\n"
        "Le script compare le résidu studentisé global à deux surprises locales : "
        "voisinage euclidien et voisinage induit par la cométrique Ridge.\n",
        encoding="utf-8",
    )

    package_path = args.output_dir / "california_local_surprise_experiment.zip"
    with zipfile.ZipFile(package_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in [
            Path(__file__), readme, args.data, detection_path, group_path, test_path,
            point_path, summary_path,
            args.output_dir / "california_local_surprise_detection.png",
            args.output_dir / "california_local_surprise_passport.png",
            args.output_dir / "california_local_surprise_weights.png",
        ]:
            if path.exists():
                archive.write(path, arcname=path.name)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
