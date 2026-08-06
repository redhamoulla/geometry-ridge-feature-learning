# Data

## Public data

The repository includes `Concrete_Data_Yeh.csv`, used by the public robustness experiment.

California Housing is public but is not duplicated in Git. The launcher uses scikit-learn to fetch it or accepts an explicit CSV through the experiment script.

## Private SeLoger table

The SeLoger source table is not redistributed. Private experiments expect a CSV path supplied through `--seloger-csv` or the `SELOGER_CSV` environment variable.

Expected columns include:

```text
number, codeinsee, codepostal, cp, etage, idagence, idannonce, idtiers,
idtypechauffage, idtypecommerce, idtypecuisine,
idtypepublicationsourcecouplage, naturebien, nb_chambres, nb_photos,
nb_pieces, position, prix, si_balcon, si_sdEau, si_sdbain, surface,
typedebien, ville
```

Do not commit the private CSV, derived personal data, or credentials.
