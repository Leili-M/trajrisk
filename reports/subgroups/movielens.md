# Section 8: subgroups and user bootstrap, movielens (DiffRec)

Judge fitted on valid_risk, evaluated on test. Mean over runs: ml_base, ml_seed1, ml_seed2, ml_seed3, ml_seed4.
Target popularity: tail <= 53.5 < mid <= 367.7 < head (training positives; 50% / 90% quantiles of catalog items).

| Kind | Group | Share of cases | Miss rate | AUROC controls3 | Gain controls3 + lists over controls3 | Gain controls + lists over controls | Gain combined over controls |
|---|---|---|---|---|---|---|---|
| history | 1 | 0.006 | 0.828 | 0.549 | +0.008 | +0.001 | -0.002 |
| history | 2-3 | 0.013 | 0.825 | 0.542 | +0.003 | +0.002 | +0.004 |
| history | 4-9 | 0.039 | 0.820 | 0.551 | +0.009 | +0.003 | +0.009 |
| history | 10-29 | 0.118 | 0.829 | 0.556 | +0.009 | +0.003 | +0.002 |
| history | 30+ | 0.824 | 0.922 | 0.580 | -0.000 | -0.000 | +0.000 |
| target popularity | tail | 0.094 | 1.000 | 0.403 | +0.049 | -0.011 | -0.024 |
| target popularity | mid | 0.471 | 0.978 | 0.527 | +0.009 | +0.000 | -0.004 |
| target popularity | head | 0.435 | 0.807 | 0.563 | +0.001 | +0.000 | +0.001 |

## Paired user bootstrap (200 resamples), 95% interval of the gain

| Run | controls3 + lists over controls3 | controls + lists over controls | combined over controls |
|---|---|---|---|
| ml_base | [-0.001, +0.001] | [-0.001, +0.000] | [-0.000, +0.001] |
| ml_seed1 | [-0.000, +0.002] | [-0.000, +0.001] | [+0.000, +0.002] |
| ml_seed2 | [-0.001, +0.001] | [-0.001, -0.000] | [-0.001, +0.000] |
| ml_seed3 | [-0.001, +0.001] | [-0.001, +0.000] | [-0.001, +0.001] |
| ml_seed4 | [-0.000, +0.002] | [-0.000, +0.002] | [+0.001, +0.003] |
