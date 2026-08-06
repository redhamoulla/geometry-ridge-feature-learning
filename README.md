# Conditional Geometric Register of Observations

Research paper and simulation code for **Conditional Geometric Register of Observations — From Ridge to Feature Learning: Invariants, Diagnostics, and Dynamics**.

The central per-observation datum is

\[
\mathfrak D_{i\mid S}=(G_S,\Gamma_i,\alpha_i),
\]

which separates the geometry already learned from the observation operator and the cotangent effort induced by the label.

## Layout

```text
paper/                 Complete English LaTeX manuscript, split into sections
src/geometric_register Core register and bounded-geometric ridge utilities
experiments/           Factorization, nonlinear, robustness, data-quality, SeLoger
tests/                 Unit tests
```

## Install and test

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
pytest
```

## Run experiments

```bash
python experiments/factorization.py
python experiments/nonlinear.py
python experiments/robustness.py
python experiments/data_quality.py
python experiments/seloger.py '/path/to/selogerdata(1).csv'
```

The private SeLoger CSV is not distributed. The personal website is not included in the manuscript or repository documentation.
