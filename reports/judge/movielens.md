# Sections 7-8: judge on movielens (DiffRec)

Judge fitted on **valid_risk**, evaluated once on **test**. Mean over runs with 95% t-interval.
Runs: ml_base (test 110,457 cases, miss 0.905), ml_seed1 (test 110,457 cases, miss 0.905), ml_seed2 (test 110,457 cases, miss 0.905), ml_seed3 (test 110,457 cases, miss 0.906), ml_seed4 (test 110,457 cases, miss 0.905)

## Primary endpoint: AUROC gain (success if >= 0.02)

| Comparison | Gain [95% CI] |
|---|---|
| combined over controls3 | 0.002 [0.001, 0.003] |
| controls3 + trajectory over controls3 | 0.001 [0.000, 0.002] |
| combined over controls | 0.001 [-0.000, 0.002] |
| CV inside test (earlier protocol): controls3 + trajectory over controls3 | 0.001 [0.000, 0.001] |

## Predictor sets (test)

| Predictors | AUROC | AUPRC | Brier | AURC (miss) | AURC (1-NDCG) |
|---|---|---|---|---|---|
| controls3 | 0.628 [0.626, 0.630] | 0.934 [0.933, 0.935] | 0.084 [0.084, 0.084] | 0.865 [0.864, 0.866] | 0.932 [0.932, 0.933] |
| controls | 0.629 [0.627, 0.631] | 0.934 [0.934, 0.935] | 0.084 [0.084, 0.084] | 0.864 [0.863, 0.865] | 0.932 [0.931, 0.932] |
| trajectory | 0.620 [0.616, 0.623] | 0.932 [0.931, 0.933] | 0.085 [0.084, 0.085] | 0.869 [0.868, 0.870] | 0.934 [0.933, 0.935] |
| controls3 + trajectory | 0.629 [0.628, 0.630] | 0.934 [0.933, 0.935] | 0.084 [0.084, 0.084] | 0.863 [0.863, 0.864] | 0.932 [0.931, 0.932] |
| combined | 0.630 [0.628, 0.631] | 0.934 [0.934, 0.935] | 0.084 [0.084, 0.084] | 0.863 [0.862, 0.863] | 0.932 [0.931, 0.932] |

Failure prevalence (AUPRC of a random score): 0.905. References (not deployable): AURC miss random 0.906 / oracle 0.682; AURC 1-NDCG random 0.953 / oracle 0.823.

## Untrained single signals (test, mean over runs)

| Signal | AUROC | AUPRC | AURC (miss) | AURC (1-NDCG) |
|---|---|---|---|---|
| `u_set_r0.25` | 0.459 | 0.895 | 0.911 | 0.957 |
| `u_rank_r0.25` | 0.437 | 0.887 | 0.919 | 0.961 |
| `score_change_unif` | 0.380 | 0.871 | 0.932 | 0.967 |
| `score_change_first` | 0.381 | 0.871 | 0.932 | 0.967 |
| `cons_set` | 0.458 | 0.895 | 0.911 | 0.957 |
| `small_margin` | 0.468 | 0.898 | 0.914 | 0.958 |

Predictor sets: **controls3** = log_history, small_margin, history_popularity; **controls** = log_history, small_margin, history_popularity, rec_popularity, top_margin, entropy_t0.1; **trajectory** = u_set_r0.25, u_rank_r0.25, score_change_unif, score_change_first, cons_set, cons_rank, reversal_set; **controls3 + trajectory** = log_history, small_margin, history_popularity, u_set_r0.25, u_rank_r0.25, score_change_unif, score_change_first, cons_set, cons_rank, reversal_set; **combined** = log_history, small_margin, history_popularity, rec_popularity, top_margin, entropy_t0.1, u_set_r0.25, u_rank_r0.25, score_change_unif, score_change_first, cons_set, cons_rank, reversal_set

Reliability tables of the combined judge: `reliability_<run>.csv`.
