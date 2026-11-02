# Sections 7-8: judge on amazon_video_games (DiffRec)

Judge fitted on **valid_risk**, evaluated once on **test**. Mean over runs with 95% t-interval.
Runs: vg_base (test 26,350 cases, miss 0.930), vg_seed1 (test 26,350 cases, miss 0.928), vg_seed2 (test 26,350 cases, miss 0.928), vg_seed3 (test 26,350 cases, miss 0.931), vg_seed4 (test 26,350 cases, miss 0.929)

## Primary endpoint: AUROC gain (success if >= 0.02)

| Comparison | Gain [95% CI] |
|---|---|
| combined over controls3 | 0.065 [0.059, 0.072] |
| controls3 + trajectory over controls3 | 0.053 [0.049, 0.057] |
| combined over controls | 0.020 [0.016, 0.024] |
| CV inside test (earlier protocol): controls3 + trajectory over controls3 | 0.055 [0.052, 0.059] |

## Predictor sets (test)

| Predictors | AUROC | AUPRC | Brier | AURC (miss) | AURC (1-NDCG) |
|---|---|---|---|---|---|
| controls3 | 0.597 [0.592, 0.602] | 0.948 [0.946, 0.950] | 0.065 [0.064, 0.067] | 0.907 [0.905, 0.909] | 0.947 [0.947, 0.948] |
| controls | 0.642 [0.638, 0.646] | 0.955 [0.953, 0.956] | 0.065 [0.063, 0.066] | 0.892 [0.890, 0.894] | 0.937 [0.936, 0.938] |
| trajectory | 0.603 [0.600, 0.606] | 0.950 [0.948, 0.951] | 0.065 [0.064, 0.066] | 0.907 [0.905, 0.908] | 0.947 [0.946, 0.948] |
| controls3 + trajectory | 0.649 [0.647, 0.652] | 0.956 [0.954, 0.957] | 0.064 [0.063, 0.066] | 0.892 [0.889, 0.894] | 0.938 [0.937, 0.940] |
| combined | 0.662 [0.658, 0.666] | 0.958 [0.956, 0.959] | 0.064 [0.063, 0.065] | 0.886 [0.884, 0.887] | 0.934 [0.932, 0.935] |

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

Predictor sets: **controls3** = log_history, small_margin, history_popularity; **controls** = log_history, small_margin, history_popularity, rec_popularity, top_margin, entropy_t1; **trajectory** = u_set_r0.75, u_rank_unif, score_change_unif, score_change_first, cons_set, cons_rank, reversal_set; **controls3 + trajectory** = log_history, small_margin, history_popularity, u_set_r0.75, u_rank_unif, score_change_unif, score_change_first, cons_set, cons_rank, reversal_set; **combined** = log_history, small_margin, history_popularity, rec_popularity, top_margin, entropy_t1, u_set_r0.75, u_rank_unif, score_change_unif, score_change_first, cons_set, cons_rank, reversal_set

Reliability tables of the combined judge: `reliability_<run>.csv`.
