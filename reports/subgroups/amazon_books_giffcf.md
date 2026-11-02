# Section 8: subgroups and user bootstrap, amazon_books (GiffCF)

Judge fitted on valid_risk, evaluated on test. Mean over runs: bk_giff_s1, bk_giff_s2, bk_giff_s3.
Target popularity: tail <= 32 < mid <= 90 < head (training positives; 50% / 90% quantiles of catalog items).

| Kind | Group | Share of cases | Miss rate | AUROC controls3 | Gain controls3 + lists over controls3 | Gain controls + lists over controls | Gain combined over controls |
|---|---|---|---|---|---|---|---|
| history | 1 | 0.175 | 0.896 | 0.573 | +0.008 | -0.001 | +0.064 |
| history | 2-3 | 0.276 | 0.869 | 0.589 | +0.004 | +0.003 | +0.028 |
| history | 4-9 | 0.299 | 0.862 | 0.594 | -0.000 | +0.000 | +0.017 |
| history | 10-29 | 0.170 | 0.864 | 0.591 | +0.008 | -0.001 | +0.003 |
| history | 30+ | 0.080 | 0.914 | 0.609 | +0.002 | -0.001 | -0.006 |
| target popularity | tail | 0.212 | 0.899 | 0.587 | +0.012 | +0.008 | +0.048 |
| target popularity | mid | 0.350 | 0.880 | 0.595 | +0.014 | +0.007 | +0.046 |
| target popularity | head | 0.438 | 0.857 | 0.549 | +0.035 | +0.006 | +0.024 |

## Paired user bootstrap (200 resamples), 95% interval of the gain

| Run | controls3 + lists over controls3 | controls + lists over controls | combined over controls |
|---|---|---|---|
| bk_giff_s1 | [+0.017, +0.023] | [+0.004, +0.006] | [+0.022, +0.027] |
| bk_giff_s2 | [+0.018, +0.024] | [+0.007, +0.009] | [+0.054, +0.060] |
| bk_giff_s3 | [+0.030, +0.037] | [+0.006, +0.008] | [+0.024, +0.028] |
