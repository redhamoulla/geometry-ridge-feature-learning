# Reproducibility

## Public suites

```bash
python run_experiments.py --suite quick
python run_experiments.py --suite public
```

`quick` runs the compact algebraic checks. `public` additionally runs the nonlinear, California Housing, and Concrete experiments.

Generated files are written under `results/generated/` and ignored by Git.

## Private SeLoger suite

```bash
python run_experiments.py \
  --suite seloger \
  --seloger-csv '/absolute/path/to/selogerdata(1).csv'
```

The launcher sets `SELOGER_CSV` for the private scripts.

## Reference outputs

`results/reference/` contains selected JSON summaries from the manuscript archive. They support integrity checks and comparison, but do not substitute for rerunning the public experiments.

## Determinism

The scripts fix their random seeds. Numerical differences can still occur across BLAS, PyTorch, scikit-learn, and operating-system versions. The CI intentionally runs only lightweight smoke checks.

## Known limitations

- SeLoger cannot be reproduced without the private source table.
- Full nonlinear experiments can be computationally expensive.
- The bounded-geometric estimator is a prototype used to validate structural mechanisms, not a claim of universal superiority over established robust regression methods.
