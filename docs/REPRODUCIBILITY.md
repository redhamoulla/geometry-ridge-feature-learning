# Reproducibility

## Public checks

```bash
python run_experiments.py --suite quick
python run_experiments.py --suite public
```

`quick` runs the compact algebraic experiments. `public` additionally runs the nonlinear, Concrete, and California Housing experiments. Full nonlinear runs can be computationally expensive.

## Private SeLoger suite

```bash
python run_experiments.py --suite seloger --seloger-csv '/absolute/path/to/selogerdata.csv'
```

The source table is not redistributed. The launcher sets `SELOGER_CSV` for the private scripts.

## Determinism

Experiment scripts fix random seeds. Small numerical differences can occur across BLAS, PyTorch, scikit-learn, and operating-system versions.
