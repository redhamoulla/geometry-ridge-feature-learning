# Conditional Geometric Register of Observations

[![CI](https://github.com/redhamoulla/geometry-ridge-feature-learning/actions/workflows/ci.yml/badge.svg)](https://github.com/redhamoulla/geometry-ridge-feature-learning/actions/workflows/ci.yml)

Research paper and reproducible experiments for:

> **Conditional Geometric Register of Observations — From Ridge to Feature Learning: Invariants, Diagnostics, and Dynamics**

The repository studies a conditional, non-scalar geometric description of each observation,

\[
\mathfrak D_{i\mid S}=(G_S,\Gamma_i,\alpha_i),
\]

where `G_S` is the geometry already induced by a context set, `Γ_i` is the positive observation operator contributed by a new point, and `α_i` is its cotangent effort. The paper shows how several scalar diagnostics arise as projections of this richer object and studies its evolution under feature learning.

## Repository layout

```text
.
├── paper/                    English paper and LaTeX sources
├── experiments/
│   ├── factorization/        Exact factorization and orbit-signature checks
│   ├── nonlinear/            Dynamic-register and feature-learning experiments
│   ├── robustness/           Public bounded-geometric robustness benchmark
│   ├── data_quality/         Local conditional-surprise experiment
│   └── private/              SeLoger scripts; private CSV not redistributed
├── data/public/              Public California Housing and Concrete tables
├── results/reference/        Reference CSV/JSON outputs
├── docs/                     Reproducibility notes
├── tests/                    Lightweight smoke tests
├── run_experiments.py        Unified experiment launcher
└── experiment_manifest.json  Script and data inventory
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

A Conda/Mamba environment is also provided:

```bash
mamba env create -f environment.yml
mamba activate geometric-register-v7
```

## Reproduce the experiments

Compact deterministic checks:

```bash
python run_experiments.py --suite quick
```

All public experiments:

```bash
python run_experiments.py --suite public
```

Private SeLoger experiments:

```bash
python run_experiments.py --suite seloger --seloger-csv '/path/to/selogerdata(1).csv'
```

The SeLoger CSV is not redistributed. Its expected schema is documented in [`data/README.md`](data/README.md).

## Paper

- Compiled manuscript: [`paper/conditional_geometric_register_of_observations_v7_en.pdf`](paper/conditional_geometric_register_of_observations_v7_en.pdf)
- LaTeX sources: [`paper/src/`](paper/src/)

## Reproducibility scope

The factorization, nonlinear, California Housing, and Concrete experiments use public or synthetic data. SeLoger experiments require the private source table and therefore cannot be reproduced from this repository alone. Detailed commands and limitations are documented in [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Citation

Citation metadata are provided in [`CITATION.cff`](CITATION.cff).

## License

No software or document license has been selected. Default copyright rules therefore apply.
