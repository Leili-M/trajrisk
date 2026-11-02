# Section 8: subgroups and user bootstrap, base (DiffRec)

Judge fitted on valid_risk, evaluated on test. Mean over runs: hm_seed42, hm_seed1, hm_seed2, hm_seed3, hm_seed4.
Target popularity: tail <= 8 < mid <= 25 < head (training positives; 50% / 90% quantiles of catalog items).

| Kind | Group | Share of cases | Miss rate | AUROC controls3 | Gain controls3 + lists over controls3 | Gain controls + lists over controls | Gain combined over controls |
|---|---|---|---|---|---|---|---|
| history | 1 | 0.110 | 0.972 | 0.633 | +0.001 | -0.004 | -0.006 |
| history | 2-3 | 0.089 | 0.977 | 0.620 | +0.005 | -0.004 | -0.009 |
| history | 4-9 | 0.114 | 0.978 | 0.695 | +0.005 | +0.000 | -0.001 |
| history | 10-29 | 0.150 | 0.982 | 0.669 | -0.010 | -0.005 | -0.004 |
| history | 30+ | 0.536 | 0.990 | 0.705 | -0.006 | -0.000 | +0.001 |
| target popularity | tail | 0.201 | 1.000 | 0.304 | +0.038 | +0.028 | +0.016 |
| target popularity | mid | 0.314 | 1.000 | 0.181 | +0.016 | +0.004 | +0.016 |
| target popularity | head | 0.485 | 0.968 | 0.641 | -0.001 | -0.002 | -0.004 |

## Paired user bootstrap (200 resamples), 95% interval of the gain

| Run | controls3 + lists over controls3 | controls + lists over controls | combined over controls |
|---|---|---|---|
| hm_seed42 | [-0.007, +0.002] | [-0.005, +0.001] | [-0.010, -0.001] |
| hm_seed1 | [-0.017, +0.003] | [-0.006, +0.003] | [-0.011, +0.001] |
| hm_seed2 | [-0.010, +0.004] | [-0.004, +0.003] | [-0.005, +0.004] |
| hm_seed3 | [-0.008, +0.007] | [-0.004, +0.003] | [-0.005, +0.002] |
| hm_seed4 | [-0.015, +0.001] | [-0.011, -0.003] | [-0.012, -0.003] |
