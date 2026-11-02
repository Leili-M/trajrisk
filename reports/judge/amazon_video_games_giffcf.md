# Sections 7-8: judge on amazon_video_games (GiffCF)

Judge fitted on **valid_risk**, evaluated once on **test**. Mean over runs with 95% t-interval.
Runs: vg_giff_s1 (test 26,350 cases, miss 0.936), vg_giff_s2 (test 26,350 cases, miss 0.936), vg_giff_s3 (test 26,350 cases, miss 0.930)

## Primary endpoint: AUROC gain (success if >= 0.02)

| Comparison | Gain [95% CI] |
|---|---|
| combined over controls3 | 0.083 [0.043, 0.123] |
| controls3 + trajectory over controls3 | 0.017 [-0.027, 0.062] |
| combined over controls | 0.004 [-0.009, 0.017] |
| controls3 + lists over controls3 | 0.006 [-0.013, 0.026] |
| controls + lists over controls | 0.000 [-0.004, 0.004] |
| CV inside test (earlier protocol): controls3 + trajectory over controls3 | 0.020 [-0.019, 0.058] |

## Predictor sets (test)

| Predictors | AUROC | AUPRC | Brier | AURC (miss) | AURC (1-NDCG) |
|---|---|---|---|---|---|
| controls3 | 0.596 [0.561, 0.630] | 0.951 [0.942, 0.960] | 0.062 [0.054, 0.069] | 0.914 [0.912, 0.915] | 0.950 [0.948, 0.953] |
| controls | 0.675 [0.608, 0.742] | 0.963 [0.948, 0.977] | 0.061 [0.051, 0.070] | 0.893 [0.886, 0.901] | 0.936 [0.930, 0.941] |
| trajectory | 0.560 [0.485, 0.636] | 0.944 [0.922, 0.966] | 0.062 [0.054, 0.070] | 0.922 [0.916, 0.928] | 0.954 [0.948, 0.959] |
| controls3 + trajectory | 0.613 [0.539, 0.687] | 0.953 [0.937, 0.969] | 0.061 [0.053, 0.070] | 0.909 [0.898, 0.920] | 0.947 [0.938, 0.956] |
| combined | 0.679 [0.606, 0.752] | 0.963 [0.949, 0.978] | 0.060 [0.051, 0.070] | 0.892 [0.881, 0.902] | 0.935 [0.927, 0.942] |
| controls3 + lists | 0.602 [0.548, 0.656] | 0.951 [0.939, 0.962] | 0.062 [0.054, 0.070] | 0.911 [0.905, 0.918] | 0.949 [0.942, 0.955] |
| controls + lists | 0.675 [0.606, 0.745] | 0.962 [0.948, 0.977] | 0.060 [0.051, 0.070] | 0.893 [0.885, 0.901] | 0.935 [0.929, 0.942] |

Failure prevalence (AUPRC of a random score): 0.934. References (not deployable): AURC miss random 0.936 / oracle 0.754; AURC 1-NDCG random 0.963 / oracle 0.842.

## Untrained single signals (test, mean over runs)

| Signal | AUROC | AUPRC | AURC (miss) | AURC (1-NDCG) |
|---|---|---|---|---|
| `u_set_unif` | 0.496 | 0.933 | 0.936 | 0.963 |
| `u_rank_r0.75` | 0.520 | 0.937 | 0.929 | 0.958 |
| `score_change_unif` | 0.449 | 0.925 | 0.943 | 0.968 |
| `score_change_first` | 0.450 | 0.925 | 0.943 | 0.968 |
| `cons_set` | 0.494 | 0.933 | 0.936 | 0.963 |
| `small_margin` | 0.539 | 0.940 | 0.924 | 0.956 |

Predictor sets: **controls3** = log_history, small_margin, history_popularity; **controls** = log_history, small_margin, history_popularity, rec_popularity, top_margin, entropy_t1; **trajectory** = u_set_unif, u_rank_r0.75, score_change_unif, score_change_first, cons_set, cons_rank, reversal_set; **controls3 + trajectory** = log_history, small_margin, history_popularity, u_set_unif, u_rank_r0.75, score_change_unif, score_change_first, cons_set, cons_rank, reversal_set; **combined** = log_history, small_margin, history_popularity, rec_popularity, top_margin, entropy_t1, u_set_unif, u_rank_r0.75, score_change_unif, score_change_first, cons_set, cons_rank, reversal_set; **controls3 + lists** = log_history, small_margin, history_popularity, u_set_unif, u_rank_r0.75, cons_set, cons_rank, reversal_set; **controls + lists** = log_history, small_margin, history_popularity, rec_popularity, top_margin, entropy_t1, u_set_unif, u_rank_r0.75, cons_set, cons_rank, reversal_set

Reliability tables of the combined judge: `reliability_<run>.csv`.
