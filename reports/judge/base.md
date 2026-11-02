# Sections 7-8: judge on base (DiffRec)

Judge fitted on **valid_risk**, evaluated once on **test**. Mean over runs with 95% t-interval.
Runs: hm_seed42 (test 60,458 cases, miss 0.984), hm_seed1 (test 60,458 cases, miss 0.985), hm_seed2 (test 60,458 cases, miss 0.984), hm_seed3 (test 60,458 cases, miss 0.985), hm_seed4 (test 60,458 cases, miss 0.984)

## Primary endpoint: AUROC gain (success if >= 0.02)

| Comparison | Gain [95% CI] |
|---|---|
| combined over controls3 | 0.003 [-0.003, 0.009] |
| controls3 + trajectory over controls3 | -0.008 [-0.015, -0.001] |
| combined over controls | -0.004 [-0.008, -0.001] |
| CV inside test (earlier protocol): controls3 + trajectory over controls3 | -0.004 [-0.004, -0.003] |

## Predictor sets (test)

| Predictors | AUROC | AUPRC | Brier | AURC (miss) | AURC (1-NDCG) |
|---|---|---|---|---|---|
| controls3 | 0.701 [0.694, 0.707] | 0.993 [0.992, 0.993] | 0.015 [0.015, 0.016] | 0.973 [0.971, 0.974] | 0.986 [0.985, 0.986] |
| controls | 0.708 [0.700, 0.716] | 0.993 [0.992, 0.993] | 0.015 [0.015, 0.016] | 0.972 [0.971, 0.974] | 0.985 [0.985, 0.986] |
| trajectory | 0.647 [0.628, 0.666] | 0.990 [0.989, 0.991] | 0.015 [0.015, 0.016] | 0.976 [0.974, 0.978] | 0.987 [0.986, 0.989] |
| controls3 + trajectory | 0.692 [0.688, 0.697] | 0.992 [0.992, 0.992] | 0.015 [0.015, 0.016] | 0.973 [0.971, 0.974] | 0.986 [0.985, 0.986] |
| combined | 0.704 [0.695, 0.712] | 0.993 [0.992, 0.993] | 0.015 [0.015, 0.016] | 0.972 [0.971, 0.974] | 0.985 [0.985, 0.986] |

Failure prevalence (AUPRC of a random score): 0.984. References (not deployable): AURC miss random 0.984 / oracle 0.919; AURC 1-NDCG random 0.992 / oracle 0.955.

## Untrained single signals (test, mean over runs)

| Signal | AUROC | AUPRC | AURC (miss) | AURC (1-NDCG) |
|---|---|---|---|---|
| `u_set_unif` | 0.457 | 0.983 | 0.989 | 0.994 |
| `u_rank_unif` | 0.444 | 0.982 | 0.988 | 0.994 |
| `score_change_unif` | 0.366 | 0.978 | 0.991 | 0.995 |
| `score_change_first` | 0.370 | 0.978 | 0.990 | 0.995 |
| `cons_set` | 0.454 | 0.983 | 0.989 | 0.994 |
| `small_margin` | 0.443 | 0.982 | 0.987 | 0.994 |

Predictor sets: **controls3** = log_history, small_margin, history_popularity; **controls** = log_history, small_margin, history_popularity, rec_popularity, top_margin, entropy_t0.1; **trajectory** = u_set_unif, u_rank_unif, score_change_unif, score_change_first, cons_set, cons_rank, reversal_set; **controls3 + trajectory** = log_history, small_margin, history_popularity, u_set_unif, u_rank_unif, score_change_unif, score_change_first, cons_set, cons_rank, reversal_set; **combined** = log_history, small_margin, history_popularity, rec_popularity, top_margin, entropy_t0.1, u_set_unif, u_rank_unif, score_change_unif, score_change_first, cons_set, cons_rank, reversal_set

Reliability tables of the combined judge: `reliability_<run>.csv`.
