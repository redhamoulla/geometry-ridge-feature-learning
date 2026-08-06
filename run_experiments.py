#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
QUICK = [
    ROOT / "experiments/factorization/test_nonscalar_orbit_collision.py",
    ROOT / "experiments/factorization/test_orbit_signature_complementarity.py",
]
PUBLIC = QUICK + [
    ROOT / "experiments/factorization/test_factorization_and_dynamic_strictness.py",
    ROOT / "experiments/nonlinear/test_nonlinear_dynamic_register.py",
    ROOT / "experiments/nonlinear/test_nonlinear_functional_drift.py",
    ROOT / "experiments/nonlinear/test_functional_drift_controls.py",
    ROOT / "experiments/nonlinear/test_stagewise_dynamic_register.py",
    ROOT / "experiments/robustness/test_public_bgr_concrete.py",
    ROOT / "experiments/data_quality/test_local_conditional_surprise_california.py",
]
SELOGER = [
    ROOT / "experiments/private/select_seloger_robust_hyperparameters.py",
    ROOT / "experiments/private/test_bounded_geometric_seloger.py",
    ROOT / "experiments/private/test_schweppe_baseline_seloger.py",
    ROOT / "experiments/private/test_seloger_baseline_mass_sensitivity.py",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=["quick", "public", "seloger"], default="quick")
    parser.add_argument("--seloger-csv", type=Path)
    args = parser.parse_args()

    env = os.environ.copy()
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("OMP_NUM_THREADS", "1")
    env["CONCRETE_CSV"] = str(ROOT / "data/public/Concrete_Data_Yeh.csv")

    scripts = QUICK if args.suite == "quick" else PUBLIC if args.suite == "public" else SELOGER
    if args.suite == "seloger":
        if args.seloger_csv is None or not args.seloger_csv.expanduser().is_file():
            parser.error("--seloger-csv must point to the private CSV")
        env["SELOGER_CSV"] = str(args.seloger_csv.expanduser().resolve())

    for script in scripts:
        command = [sys.executable, str(script)]
        if script.name == "test_local_conditional_surprise_california.py":
            output = ROOT / "results/generated/california"
            output.mkdir(parents=True, exist_ok=True)
            command += [
                "--data",
                str(ROOT / "data/public/california_housing_raw.csv"),
                "--output-dir",
                str(output),
            ]
        print("+", " ".join(command), flush=True)
        subprocess.run(command, cwd=script.parent, env=env, check=True)


if __name__ == "__main__":
    main()
