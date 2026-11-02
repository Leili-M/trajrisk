# Sections 7-8: judge on movielens (DiffRec)

Judge fitted on **valid_risk**, evaluated once on **test**. Mean over runs with 95% t-interval.
Runs: ml_base (test 110,457 cases, miss 0.905), ml_seed1 (test 110,457 cases, miss 0.905), ml_seed2 (test 110,457 cases, miss 0.905), ml_seed3 (test 110,457 cases, miss 0.906), ml_seed4 (test 110,457 cases, miss 0.905)

## Primary endpoint: AUROC gain (success if >= 0.02)

| Comparison | Gain [95% CI] |
|---|---|
| combined over controls3 | 0.002 [0.001, 0.003] |
| controls3 + trajectory over controls3 | 0.001 [0.000, 0.002] |
| combined over controls | 0.001 [-0.000, 0.002] |
| controls3 + lists over controls3 | 0.000 [-0.000, 0.001] |
| controls + lists over controls | -0.000 [-0.001, 0.001] |
| controls3 + sampling2 over controls3 | 0.001 [0.000, 0.002] |
| controls3 + sampling5 over controls3 | 0.001 [0.001, 0.002] |
| controls3 + sampling10 over controls3 | 0.001 [0.001, 0.002] |
| controls + sampling2 over controls | 0.001 [-0.000, 0.001] |
| controls + sampling5 over controls | 0.001 [0.000, 0.001] |
| controls + sampling10 over controls | 0.001 [0.000, 0.001] |
| controls3 + masking over controls3 | -0.002 [-0.002, -0.002] |
| controls + masking over controls | -0.002 [-0.002, -0.001] |
| controls + lists + sampling10 over controls + lists | 0.000 [0.000, 0.001] |
| controls + lists + masking over controls + lists | -0.002 [-0.002, -0.001] |
| all over controls | -0.002 [-0.003, -0.001] |
| CV inside test (earlier protocol): controls3 + trajectory over controls3 | 0.001 [0.000, 0.001] |

## Predictor sets (test)

| Predictors | AUROC | AUPRC | Brier | AURC (miss) | AURC (1-NDCG) |
|---|---|---|---|---|---|
| controls3 | 0.628 [0.626, 0.630] | 0.934 [0.933, 0.935] | 0.084 [0.084, 0.084] | 0.865 [0.864, 0.866] | 0.932 [0.932, 0.933] |
| controls | 0.629 [0.627, 0.631] | 0.934 [0.934, 0.935] | 0.084 [0.084, 0.084] | 0.864 [0.863, 0.865] | 0.932 [0.931, 0.932] |
| trajectory | 0.620 [0.616, 0.623] | 0.932 [0.931, 0.933] | 0.085 [0.084, 0.085] | 0.869 [0.868, 0.870] | 0.934 [0.933, 0.935] |
| controls3 + trajectory | 0.629 [0.628, 0.630] | 0.934 [0.933, 0.935] | 0.084 [0.084, 0.084] | 0.863 [0.863, 0.864] | 0.932 [0.931, 0.932] |
| combined | 0.630 [0.628, 0.631] | 0.934 [0.934, 0.935] | 0.084 [0.084, 0.084] | 0.863 [0.862, 0.863] | 0.932 [0.931, 0.932] |
| controls3 + lists | 0.628 [0.627, 0.630] | 0.934 [0.933, 0.935] | 0.084 [0.084, 0.084] | 0.865 [0.864, 0.865] | 0.932 [0.932, 0.933] |
| controls + lists | 0.629 [0.627, 0.631] | 0.934 [0.934, 0.935] | 0.084 [0.084, 0.084] | 0.864 [0.863, 0.864] | 0.932 [0.931, 0.932] |
| controls3 + sampling2 | 0.629 [0.627, 0.630] | 0.934 [0.933, 0.935] | 0.084 [0.084, 0.084] | 0.863 [0.863, 0.864] | 0.932 [0.931, 0.932] |
| controls3 + sampling5 | 0.629 [0.627, 0.631] | 0.934 [0.933, 0.935] | 0.084 [0.084, 0.084] | 0.863 [0.862, 0.864] | 0.932 [0.931, 0.932] |
| controls3 + sampling10 | 0.629 [0.627, 0.631] | 0.934 [0.933, 0.935] | 0.084 [0.084, 0.084] | 0.863 [0.862, 0.864] | 0.932 [0.931, 0.932] |
| controls + sampling2 | 0.630 [0.628, 0.632] | 0.934 [0.934, 0.935] | 0.084 [0.084, 0.084] | 0.863 [0.862, 0.864] | 0.932 [0.931, 0.932] |
| controls + sampling5 | 0.630 [0.628, 0.632] | 0.934 [0.934, 0.935] | 0.084 [0.084, 0.084] | 0.863 [0.862, 0.864] | 0.932 [0.931, 0.932] |
| controls + sampling10 | 0.630 [0.628, 0.632] | 0.934 [0.934, 0.935] | 0.084 [0.084, 0.084] | 0.863 [0.862, 0.864] | 0.931 [0.931, 0.932] |
| controls3 + masking | 0.626 [0.624, 0.628] | 0.933 [0.933, 0.934] | 0.084 [0.084, 0.084] | 0.865 [0.864, 0.866] | 0.932 [0.932, 0.933] |
| controls + masking | 0.628 [0.625, 0.630] | 0.934 [0.933, 0.935] | 0.084 [0.084, 0.084] | 0.863 [0.863, 0.864] | 0.932 [0.931, 0.932] |
| controls + lists + sampling10 | 0.630 [0.628, 0.631] | 0.934 [0.934, 0.935] | 0.084 [0.084, 0.084] | 0.863 [0.862, 0.864] | 0.932 [0.931, 0.932] |
| controls + lists + masking | 0.627 [0.626, 0.629] | 0.934 [0.933, 0.934] | 0.084 [0.084, 0.084] | 0.863 [0.863, 0.864] | 0.932 [0.931, 0.932] |
| all | 0.627 [0.625, 0.630] | 0.934 [0.933, 0.935] | 0.084 [0.084, 0.084] | 0.863 [0.862, 0.864] | 0.932 [0.931, 0.932] |

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
| `samp10_set` | 0.460 | 0.896 | 0.909 | 0.956 |
| `samp10_rank` | 0.431 | 0.886 | 0.921 | 0.961 |
| `samp10_score` | 0.380 | 0.870 | 0.932 | 0.967 |
| `mask_set` | 0.550 | 0.918 | 0.889 | 0.945 |
| `mask_rank` | 0.550 | 0.918 | 0.889 | 0.944 |

Predictor sets: **controls3** = log_history, small_margin, history_popularity; **controls** = log_history, small_margin, history_popularity, rec_popularity, top_margin, entropy_t0.1; **trajectory** = u_set_r0.25, u_rank_r0.25, score_change_unif, score_change_first, cons_set, cons_rank, reversal_set; **controls3 + trajectory** = log_history, small_margin, history_popularity, u_set_r0.25, u_rank_r0.25, score_change_unif, score_change_first, cons_set, cons_rank, reversal_set; **combined** = log_history, small_margin, history_popularity, rec_popularity, top_margin, entropy_t0.1, u_set_r0.25, u_rank_r0.25, score_change_unif, score_change_first, cons_set, cons_rank, reversal_set; **controls3 + lists** = log_history, small_margin, history_popularity, u_set_r0.25, u_rank_r0.25, cons_set, cons_rank, reversal_set; **controls + lists** = log_history, small_margin, history_popularity, rec_popularity, top_margin, entropy_t0.1, u_set_r0.25, u_rank_r0.25, cons_set, cons_rank, reversal_set; **controls3 + sampling2** = log_history, small_margin, history_popularity, samp2_set, samp2_rank, samp2_score; **controls3 + sampling5** = log_history, small_margin, history_popularity, samp5_set, samp5_rank, samp5_score; **controls3 + sampling10** = log_history, small_margin, history_popularity, samp10_set, samp10_rank, samp10_score; **controls + sampling2** = log_history, small_margin, history_popularity, rec_popularity, top_margin, entropy_t0.1, samp2_set, samp2_rank, samp2_score; **controls + sampling5** = log_history, small_margin, history_popularity, rec_popularity, top_margin, entropy_t0.1, samp5_set, samp5_rank, samp5_score; **controls + sampling10** = log_history, small_margin, history_popularity, rec_popularity, top_margin, entropy_t0.1, samp10_set, samp10_rank, samp10_score; **controls3 + masking** = log_history, small_margin, history_popularity, mask_set, mask_rank, mask_score, mask_unstable; **controls + masking** = log_history, small_margin, history_popularity, rec_popularity, top_margin, entropy_t0.1, mask_set, mask_rank, mask_score, mask_unstable; **controls + lists + sampling10** = log_history, small_margin, history_popularity, rec_popularity, top_margin, entropy_t0.1, u_set_r0.25, u_rank_r0.25, cons_set, cons_rank, reversal_set, samp10_set, samp10_rank, samp10_score; **controls + lists + masking** = log_history, small_margin, history_popularity, rec_popularity, top_margin, entropy_t0.1, u_set_r0.25, u_rank_r0.25, cons_set, cons_rank, reversal_set, mask_set, mask_rank, mask_score, mask_unstable; **all** = log_history, small_margin, history_popularity, rec_popularity, top_margin, entropy_t0.1, u_set_r0.25, u_rank_r0.25, score_change_unif, score_change_first, cons_set, cons_rank, reversal_set, samp10_set, samp10_rank, samp10_score, mask_set, mask_rank, mask_score, mask_unstable

Reliability tables of the combined judge: `reliability_<run>.csv`.
