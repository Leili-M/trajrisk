# Section 8: subgroups and user bootstrap, amazon_books (DiffRec)

Judge fitted on valid_risk, evaluated on test. Mean over runs: bk_seed42, bk_seed1, bk_seed2, bk_seed3, bk_seed4.
Target popularity: tail <= 32 < mid <= 90 < head (training positives; 50% / 90% quantiles of catalog items).

| Kind | Group | Share of cases | Miss rate | AUROC controls3 | Gain controls3 + lists over controls3 | Gain controls + lists over controls | Gain combined over controls |
|---|---|---|---|---|---|---|---|
| history | 1 | 0.175 | 0.947 | 0.700 | +0.063 | +0.008 | +0.017 |
| history | 2-3 | 0.276 | 0.927 | 0.664 | +0.064 | +0.008 | +0.018 |
| history | 4-9 | 0.299 | 0.914 | 0.639 | +0.048 | +0.008 | +0.012 |
| history | 10-29 | 0.170 | 0.908 | 0.646 | +0.029 | +0.005 | -0.001 |
| history | 30+ | 0.080 | 0.942 | 0.659 | +0.012 | +0.014 | +0.007 |
| target popularity | tail | 0.212 | 0.997 | 0.453 | +0.259 | -0.038 | -0.021 |
| target popularity | mid | 0.350 | 0.973 | 0.545 | +0.194 | -0.016 | -0.016 |
| target popularity | head | 0.438 | 0.851 | 0.605 | +0.063 | +0.013 | +0.016 |

## Paired user bootstrap (200 resamples), 95% interval of the gain

| Run | controls3 + lists over controls3 | controls + lists over controls | combined over controls |
|---|---|---|---|
| bk_seed42 | [+0.056, +0.065] | [+0.010, +0.014] | [+0.012, +0.017] |
| bk_seed1 | [+0.060, +0.069] | [+0.010, +0.014] | [+0.011, +0.015] |
| bk_seed2 | [+0.046, +0.056] | [+0.005, +0.009] | [+0.006, +0.011] |
| bk_seed3 | [+0.037, +0.046] | [+0.003, +0.006] | [+0.004, +0.009] |
| bk_seed4 | [+0.045, +0.054] | [+0.008, +0.012] | [+0.008, +0.013] |
