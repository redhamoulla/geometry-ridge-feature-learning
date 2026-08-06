#!/usr/bin/env python3
"""Unified launcher for the manuscript experiments."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "generated"

QUICK = [
    ROOT / "experiments" / "factorization" / "test_nonscalar_orbit_collision.py",
    ROOT / "experiments" / "factorization" / "test_orbit_signature_complementarity.py",
]

PUBLIC = QUICK + [
    ROOT / "experiments" / "factorization" / "test_factorization_and_dynamic_strictness.py",
    ROOT / "experiments" / "nonlinear" / "test_nonlinear_dynamic_register.py",
    ROOT / "experiments" / "nonlinear" / "test_nonlinear_functional_drift.py",
    ROOT / "experiments" / "nonlinear" / "test_functional_drift_controls.py",
    ROOT / "experiments" / "nonlinear" / "test_stagewise_dynamic_register.py",
    ROOT / "experiments" / "robustness" / "test_public_bgr_concrete.py",
    ROOT / "experiments" / "data_quality" / "test_local_conditional_surprise_california.py",
]

SELOGER = [
    ROOT / "experiments" / "private" / "select_seloger_robust_hyperparameters.py",
    ROOT / "experiments" / "private" / "test_bounded_geometric_seloger.py",
    ROOT / "experiments" / "private" / "test_schweppe_baseline_seloger.py",
    ROOT / "experiments" / "private" / "test_seloger_baseline_mass_sensitivity.py",
]


def run_script(script: Path, env: dict[str, str], extra: list[str] | None = None) -> None:
    if not script.exists():
        raise FileNotFoundError(script)
    out = RESULTS / script.stem
    out.mkdir(parents=True, exist_ok=True)
    child_env = env.copy()
    child_env["EXPERIMENT_OUTPUT_DIR"] = str(out)
    cmd = [sys.executable, str(script)]
    if extra:
        cmd.extend(extra)
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=script.parent, env=child_env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=["quick", "public", "seloger"], default="quick")
    parser.add_argument("--seloger-csv", type=Path)
    args = parser.parse_args()

    env = os.environ.copy()
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env["CONCRETE_CSV"] = str(ROOT / "data" / "public" / "Concrete_Data_Yeh.csv")

    if args.suite == "quick":
        scripts = QUICK
    elif args.suite == "public":
        scripts = PUBLIC
    else:
        if args.seloger_csv is None:
            parser.error("--seloger-csv is required for the seloger suite")
        path = args.seloger_csv.expanduser().resolve()
        if not path.is_file():
            parser.error(f"SeLoger CSV not found: {path}")
        env["SELOGER_CSV"] = str(path)
        scripts = SELOGER

    for script in scripts:
        extra = None
        if script.name == "test_local_conditional_surprise_california.py":
            extra = ["--data", str(ROOT / "data" / "public" / "california_housing_raw.csv"), "--output-dir", str(RESULTS / script.stem)]
        run_script(script, env, extra)


if __name__ == "__main__":
    main()
