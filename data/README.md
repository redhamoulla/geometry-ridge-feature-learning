# Data

## Public data

`data/public/` contains the public tables used by the repository:

- `california_housing_raw.csv`
- `Concrete_Data_Yeh.csv`

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
