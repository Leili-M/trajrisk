# Section 8: subgroups and user bootstrap, amazon_video_games (DiffRec)

Judge fitted on valid_risk, evaluated on test. Mean over runs: vg_base, vg_seed1, vg_seed2, vg_seed3, vg_seed4.
Target popularity: tail <= 12 < mid <= 69 < head (training positives; 50% / 90% quantiles of catalog items).

| Kind | Group | Share of cases | Miss rate | AUROC controls3 | Gain controls3 + lists over controls3 | Gain controls + lists over controls | Gain combined over controls |
|---|---|---|---|---|---|---|---|
| history | 1 | 0.223 | 0.934 | 0.598 | +0.062 | +0.033 | +0.026 |
| history | 2-3 | 0.341 | 0.922 | 0.611 | +0.049 | +0.018 | +0.018 |
| history | 4-9 | 0.310 | 0.926 | 0.580 | +0.055 | +0.004 | +0.005 |
| history | 10-29 | 0.106 | 0.945 | 0.577 | +0.006 | +0.008 | +0.005 |
| history | 30+ | 0.020 | 0.958 | 0.540 | +0.016 | +0.003 | +0.013 |
| target popularity | tail | 0.175 | 0.999 | 0.224 | +0.092 | -0.058 | -0.020 |
| target popularity | mid | 0.399 | 0.971 | 0.530 | +0.130 | -0.020 | -0.030 |
| target popularity | head | 0.426 | 0.861 | 0.579 | +0.046 | +0.027 | +0.029 |

## Paired user bootstrap (200 resamples), 95% interval of the gain

| Run | controls3 + lists over controls3 | controls + lists over controls | combined over controls |
|---|---|---|---|
| vg_base | [+0.038, +0.062] | [+0.017, +0.033] | [+0.015, +0.033] |
| vg_seed1 | [+0.044, +0.066] | [+0.013, +0.025] | [+0.012, +0.026] |
| vg_seed2 | [+0.039, +0.062] | [+0.012, +0.024] | [+0.010, +0.023] |
| vg_seed3 | [+0.040, +0.065] | [+0.014, +0.028] | [+0.015, +0.029] |
| vg_seed4 | [+0.036, +0.060] | [+0.010, +0.026] | [+0.010, +0.026] |
