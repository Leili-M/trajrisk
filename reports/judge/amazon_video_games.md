# Sections 7-8: judge on amazon_video_games (DiffRec)

Judge fitted on **valid_risk**, evaluated once on **test**. Mean over runs with 95% t-interval.
Runs: vg_base (test 26,350 cases, miss 0.930), vg_seed1 (test 26,350 cases, miss 0.928), vg_seed2 (test 26,350 cases, miss 0.928), vg_seed3 (test 26,350 cases, miss 0.931), vg_seed4 (test 26,350 cases, miss 0.929)

## Primary endpoint: AUROC gain (success if >= 0.02)

| Comparison | Gain [95% CI] |
|---|---|
| combined over controls3 | 0.065 [0.059, 0.072] |
| controls3 + trajectory over controls3 | 0.053 [0.049, 0.057] |
| combined over controls | 0.020 [0.016, 0.024] |
| controls3 + lists over controls3 | 0.052 [0.049, 0.055] |
| controls + lists over controls | 0.020 [0.017, 0.024] |
| controls3 + sampling2 over controls3 | 0.059 [0.049, 0.070] |
| controls3 + sampling5 over controls3 | 0.066 [0.058, 0.075] |
| controls3 + sampling10 over controls3 | 0.069 [0.061, 0.078] |
| controls + sampling2 over controls | 0.024 [0.015, 0.034] |
| controls + sampling5 over controls | 0.029 [0.022, 0.036] |
| controls + sampling10 over controls | 0.031 [0.024, 0.037] |
| controls3 + masking over controls3 | 0.028 [0.020, 0.037] |
| controls + masking over controls | 0.008 [0.005, 0.011] |
| controls + lists + sampling10 over controls + lists | 0.010 [0.005, 0.015] |
| controls + lists + masking over controls + lists | 0.003 [0.001, 0.004] |
| all over controls | 0.030 [0.024, 0.036] |
| CV inside test (earlier protocol): controls3 + trajectory over controls3 | 0.055 [0.052, 0.059] |

## Predictor sets (test)

| Predictors | AUROC | AUPRC | Brier | AURC (miss) | AURC (1-NDCG) |
|---|---|---|---|---|---|
| controls3 | 0.597 [0.592, 0.602] | 0.948 [0.946, 0.950] | 0.065 [0.064, 0.067] | 0.907 [0.905, 0.909] | 0.947 [0.947, 0.948] |
| controls | 0.642 [0.638, 0.646] | 0.955 [0.953, 0.956] | 0.065 [0.063, 0.066] | 0.892 [0.890, 0.894] | 0.937 [0.936, 0.938] |
| trajectory | 0.603 [0.600, 0.606] | 0.950 [0.948, 0.951] | 0.065 [0.064, 0.066] | 0.907 [0.905, 0.908] | 0.947 [0.946, 0.948] |
| controls3 + trajectory | 0.649 [0.647, 0.652] | 0.956 [0.954, 0.957] | 0.064 [0.063, 0.066] | 0.892 [0.889, 0.894] | 0.938 [0.937, 0.940] |
| combined | 0.662 [0.658, 0.666] | 0.958 [0.956, 0.959] | 0.064 [0.063, 0.065] | 0.886 [0.884, 0.887] | 0.934 [0.932, 0.935] |
| controls3 + lists | 0.649 [0.646, 0.652] | 0.956 [0.954, 0.958] | 0.065 [0.063, 0.066] | 0.892 [0.890, 0.895] | 0.939 [0.937, 0.940] |
| controls + lists | 0.662 [0.660, 0.665] | 0.958 [0.956, 0.959] | 0.064 [0.063, 0.065] | 0.886 [0.884, 0.888] | 0.934 [0.933, 0.935] |
| controls3 + sampling2 | 0.656 [0.646, 0.666] | 0.957 [0.953, 0.960] | 0.064 [0.063, 0.066] | 0.889 [0.888, 0.890] | 0.937 [0.936, 0.937] |
| controls3 + sampling5 | 0.663 [0.655, 0.671] | 0.958 [0.955, 0.960] | 0.064 [0.063, 0.065] | 0.887 [0.885, 0.888] | 0.935 [0.935, 0.936] |
| controls3 + sampling10 | 0.666 [0.658, 0.674] | 0.958 [0.956, 0.961] | 0.064 [0.063, 0.065] | 0.886 [0.885, 0.887] | 0.935 [0.934, 0.935] |
| controls + sampling2 | 0.666 [0.656, 0.677] | 0.958 [0.955, 0.961] | 0.064 [0.063, 0.065] | 0.885 [0.883, 0.886] | 0.933 [0.933, 0.934] |
| controls + sampling5 | 0.671 [0.663, 0.679] | 0.959 [0.957, 0.962] | 0.064 [0.063, 0.065] | 0.884 [0.882, 0.885] | 0.933 [0.932, 0.933] |
| controls + sampling10 | 0.673 [0.665, 0.680] | 0.959 [0.957, 0.962] | 0.064 [0.063, 0.065] | 0.883 [0.882, 0.885] | 0.932 [0.932, 0.933] |
| controls3 + masking | 0.625 [0.616, 0.634] | 0.953 [0.951, 0.955] | 0.065 [0.064, 0.066] | 0.899 [0.896, 0.902] | 0.942 [0.941, 0.944] |
| controls + masking | 0.650 [0.644, 0.656] | 0.956 [0.955, 0.958] | 0.064 [0.063, 0.066] | 0.889 [0.887, 0.892] | 0.936 [0.935, 0.937] |
| controls + lists + sampling10 | 0.672 [0.665, 0.679] | 0.959 [0.957, 0.962] | 0.064 [0.063, 0.065] | 0.883 [0.882, 0.885] | 0.932 [0.932, 0.933] |
| controls + lists + masking | 0.665 [0.662, 0.669] | 0.958 [0.957, 0.960] | 0.064 [0.063, 0.065] | 0.885 [0.883, 0.887] | 0.934 [0.932, 0.935] |
| all | 0.672 [0.665, 0.680] | 0.960 [0.957, 0.962] | 0.064 [0.063, 0.065] | 0.883 [0.881, 0.885] | 0.932 [0.932, 0.933] |

Failure prevalence (AUPRC of a random score): 0.929. References (not deployable): AURC miss random 0.931 / oracle 0.742; AURC 1-NDCG random 0.961 / oracle 0.837.

## Untrained single signals (test, mean over runs)

| Signal | AUROC | AUPRC | AURC (miss) | AURC (1-NDCG) |
|---|---|---|---|---|
| `u_set_r0.75` | 0.559 | 0.940 | 0.918 | 0.954 |
| `u_rank_unif` | 0.601 | 0.949 | 0.906 | 0.946 |
| `score_change_unif` | 0.531 | 0.940 | 0.929 | 0.960 |
| `score_change_first` | 0.532 | 0.940 | 0.928 | 0.960 |
| `cons_set` | 0.560 | 0.939 | 0.918 | 0.954 |
| `small_margin` | 0.549 | 0.938 | 0.916 | 0.953 |
| `samp10_set` | 0.591 | 0.946 | 0.908 | 0.949 |
| `samp10_rank` | 0.619 | 0.952 | 0.901 | 0.944 |
| `samp10_score` | 0.531 | 0.940 | 0.929 | 0.960 |
| `mask_set` | 0.542 | 0.941 | 0.925 | 0.957 |
| `mask_rank` | 0.540 | 0.941 | 0.925 | 0.957 |

Predictor sets: **controls3** = log_history, small_margin, history_popularity; **controls** = log_history, small_margin, history_popularity, rec_popularity, top_margin, entropy_t1; **trajectory** = u_set_r0.75, u_rank_unif, score_change_unif, score_change_first, cons_set, cons_rank, reversal_set; **controls3 + trajectory** = log_history, small_margin, history_popularity, u_set_r0.75, u_rank_unif, score_change_unif, score_change_first, cons_set, cons_rank, reversal_set; **combined** = log_history, small_margin, history_popularity, rec_popularity, top_margin, entropy_t1, u_set_r0.75, u_rank_unif, score_change_unif, score_change_first, cons_set, cons_rank, reversal_set; **controls3 + lists** = log_history, small_margin, history_popularity, u_set_r0.75, u_rank_unif, cons_set, cons_rank, reversal_set; **controls + lists** = log_history, small_margin, history_popularity, rec_popularity, top_margin, entropy_t1, u_set_r0.75, u_rank_unif, cons_set, cons_rank, reversal_set; **controls3 + sampling2** = log_history, small_margin, history_popularity, samp2_set, samp2_rank, samp2_score; **controls3 + sampling5** = log_history, small_margin, history_popularity, samp5_set, samp5_rank, samp5_score; **controls3 + sampling10** = log_history, small_margin, history_popularity, samp10_set, samp10_rank, samp10_score; **controls + sampling2** = log_history, small_margin, history_popularity, rec_popularity, top_margin, entropy_t1, samp2_set, samp2_rank, samp2_score; **controls + sampling5** = log_history, small_margin, history_popularity, rec_popularity, top_margin, entropy_t1, samp5_set, samp5_rank, samp5_score; **controls + sampling10** = log_history, small_margin, history_popularity, rec_popularity, top_margin, entropy_t1, samp10_set, samp10_rank, samp10_score; **controls3 + masking** = log_history, small_margin, history_popularity, mask_set, mask_rank, mask_score, mask_unstable; **controls + masking** = log_history, small_margin, history_popularity, rec_popularity, top_margin, entropy_t1, mask_set, mask_rank, mask_score, mask_unstable; **controls + lists + sampling10** = log_history, small_margin, history_popularity, rec_popularity, top_margin, entropy_t1, u_set_r0.75, u_rank_unif, cons_set, cons_rank, reversal_set, samp10_set, samp10_rank, samp10_score; **controls + lists + masking** = log_history, small_margin, history_popularity, rec_popularity, top_margin, entropy_t1, u_set_r0.75, u_rank_unif, cons_set, cons_rank, reversal_set, mask_set, mask_rank, mask_score, mask_unstable; **all** = log_history, small_margin, history_popularity, rec_popularity, top_margin, entropy_t1, u_set_r0.75, u_rank_unif, score_change_unif, score_change_first, cons_set, cons_rank, reversal_set, samp10_set, samp10_rank, samp10_score, mask_set, mask_rank, mask_score, mask_unstable

Reliability tables of the combined judge: `reliability_<run>.csv`.
