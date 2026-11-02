# Section 8: subgroups and user bootstrap, amazon_video_games (GiffCF)

Judge fitted on valid_risk, evaluated on test. Mean over runs: vg_giff_s1, vg_giff_s2, vg_giff_s3.
Target popularity: tail <= 12 < mid <= 69 < head (training positives; 50% / 90% quantiles of catalog items).

| Kind | Group | Share of cases | Miss rate | AUROC controls3 | Gain controls3 + lists over controls3 | Gain controls + lists over controls | Gain combined over controls |
|---|---|---|---|---|---|---|---|
| history | 1 | 0.223 | 0.948 | 0.603 | +0.003 | -0.003 | -0.003 |
| history | 2-3 | 0.341 | 0.928 | 0.626 | +0.002 | -0.002 | -0.002 |
| history | 4-9 | 0.310 | 0.929 | 0.581 | +0.007 | -0.001 | +0.002 |
| history | 10-29 | 0.106 | 0.934 | 0.528 | +0.018 | +0.004 | +0.011 |
| history | 30+ | 0.020 | 0.948 | 0.530 | +0.036 | +0.003 | -0.002 |
| target popularity | tail | 0.175 | 0.976 | 0.397 | +0.027 | +0.006 | +0.005 |
| target popularity | mid | 0.399 | 0.950 | 0.456 | +0.033 | +0.002 | +0.011 |
| target popularity | head | 0.426 | 0.901 | 0.668 | -0.007 | -0.001 | +0.000 |

## Paired user bootstrap (200 resamples), 95% interval of the gain

| Run | controls3 + lists over controls3 | controls + lists over controls | combined over controls |
|---|---|---|---|
| vg_giff_s1 | [+0.002, +0.013] | [-0.004, +0.003] | [-0.003, +0.003] |
| vg_giff_s2 | [+0.005, +0.020] | [-0.001, +0.005] | [+0.004, +0.016] |
| vg_giff_s3 | [-0.010, +0.004] | [-0.005, +0.002] | [-0.002, +0.006] |
