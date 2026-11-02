# Sections 7-8: judge on base (DiffRec)

Judge fitted on **valid_risk**, evaluated once on **test**. Mean over runs with 95% t-interval.
Runs: hm_seed42 (test 60,458 cases, miss 0.984), hm_seed1 (test 60,458 cases, miss 0.985), hm_seed2 (test 60,458 cases, miss 0.984), hm_seed3 (test 60,458 cases, miss 0.985), hm_seed4 (test 60,458 cases, miss 0.984)

## Primary endpoint: AUROC gain (success if >= 0.02)

| Comparison | Gain [95% CI] |
|---|---|
| combined over controls3 | 0.003 [-0.003, 0.009] |
| controls3 + trajectory over controls3 | -0.008 [-0.015, -0.001] |
| combined over controls | -0.004 [-0.008, -0.001] |
| controls3 + lists over controls3 | -0.004 [-0.008, -0.001] |
| controls + lists over controls | -0.003 [-0.006, 0.001] |
| controls3 + sampling2 over controls3 | -0.004 [-0.012, 0.004] |
| controls3 + sampling5 over controls3 | -0.003 [-0.013, 0.006] |
| controls3 + sampling10 over controls3 | -0.003 [-0.012, 0.006] |
| controls + sampling2 over controls | -0.002 [-0.006, 0.002] |
| controls + sampling5 over controls | -0.002 [-0.006, 0.002] |
| controls + sampling10 over controls | -0.002 [-0.006, 0.002] |
| controls3 + masking over controls3 | 0.006 [-0.004, 0.017] |
| controls + masking over controls | -0.000 [-0.005, 0.004] |
| controls + lists + sampling10 over controls + lists | -0.002 [-0.005, 0.001] |
| controls + lists + masking over controls + lists | 0.000 [-0.005, 0.005] |
| all over controls | -0.004 [-0.009, 0.002] |
| CV inside test (earlier protocol): controls3 + trajectory over controls3 | -0.004 [-0.004, -0.003] |

## Predictor sets (test)

| Predictors | AUROC | AUPRC | Brier | AURC (miss) | AURC (1-NDCG) |
|---|---|---|---|---|---|
| controls3 | 0.701 [0.694, 0.707] | 0.993 [0.992, 0.993] | 0.015 [0.015, 0.016] | 0.973 [0.971, 0.974] | 0.986 [0.985, 0.986] |
| controls | 0.708 [0.700, 0.716] | 0.993 [0.992, 0.993] | 0.015 [0.015, 0.016] | 0.972 [0.971, 0.974] | 0.985 [0.985, 0.986] |
| trajectory | 0.647 [0.628, 0.666] | 0.990 [0.989, 0.991] | 0.015 [0.015, 0.016] | 0.976 [0.974, 0.978] | 0.987 [0.986, 0.989] |
| controls3 + trajectory | 0.692 [0.688, 0.697] | 0.992 [0.992, 0.992] | 0.015 [0.015, 0.016] | 0.973 [0.971, 0.974] | 0.986 [0.985, 0.986] |
| combined | 0.704 [0.695, 0.712] | 0.993 [0.992, 0.993] | 0.015 [0.015, 0.016] | 0.972 [0.971, 0.974] | 0.985 [0.985, 0.986] |
| controls3 + lists | 0.696 [0.691, 0.701] | 0.992 [0.992, 0.993] | 0.015 [0.015, 0.016] | 0.972 [0.971, 0.974] | 0.986 [0.985, 0.986] |
| controls + lists | 0.705 [0.698, 0.713] | 0.993 [0.992, 0.993] | 0.015 [0.015, 0.016] | 0.972 [0.971, 0.973] | 0.985 [0.985, 0.986] |
| controls3 + sampling2 | 0.697 [0.692, 0.702] | 0.992 [0.992, 0.993] | 0.015 [0.015, 0.016] | 0.972 [0.971, 0.974] | 0.985 [0.985, 0.986] |
| controls3 + sampling5 | 0.697 [0.692, 0.703] | 0.992 [0.992, 0.993] | 0.015 [0.015, 0.016] | 0.972 [0.971, 0.974] | 0.985 [0.985, 0.986] |
| controls3 + sampling10 | 0.698 [0.692, 0.703] | 0.992 [0.992, 0.993] | 0.015 [0.015, 0.016] | 0.972 [0.971, 0.974] | 0.985 [0.985, 0.986] |
| controls + sampling2 | 0.706 [0.697, 0.715] | 0.993 [0.992, 0.993] | 0.015 [0.015, 0.016] | 0.972 [0.970, 0.974] | 0.985 [0.984, 0.986] |
| controls + sampling5 | 0.706 [0.698, 0.713] | 0.993 [0.992, 0.993] | 0.015 [0.015, 0.016] | 0.972 [0.970, 0.974] | 0.985 [0.984, 0.986] |
| controls + sampling10 | 0.706 [0.698, 0.713] | 0.993 [0.992, 0.993] | 0.015 [0.015, 0.016] | 0.972 [0.970, 0.974] | 0.985 [0.985, 0.986] |
| controls3 + masking | 0.707 [0.695, 0.719] | 0.993 [0.992, 0.993] | 0.015 [0.015, 0.016] | 0.972 [0.970, 0.974] | 0.985 [0.984, 0.986] |
| controls + masking | 0.707 [0.697, 0.718] | 0.993 [0.992, 0.993] | 0.015 [0.015, 0.016] | 0.972 [0.970, 0.974] | 0.985 [0.985, 0.986] |
| controls + lists + sampling10 | 0.703 [0.695, 0.711] | 0.993 [0.992, 0.993] | 0.015 [0.015, 0.016] | 0.972 [0.971, 0.974] | 0.985 [0.985, 0.986] |
| controls + lists + masking | 0.705 [0.695, 0.716] | 0.993 [0.992, 0.993] | 0.015 [0.015, 0.016] | 0.972 [0.971, 0.974] | 0.985 [0.985, 0.986] |
| all | 0.704 [0.694, 0.714] | 0.993 [0.992, 0.993] | 0.015 [0.015, 0.016] | 0.972 [0.971, 0.974] | 0.985 [0.985, 0.986] |

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
| `samp10_set` | 0.423 | 0.982 | 0.989 | 0.994 |
| `samp10_rank` | 0.418 | 0.982 | 0.989 | 0.994 |
| `samp10_score` | 0.362 | 0.978 | 0.991 | 0.995 |
| `mask_set` | 0.608 | 0.989 | 0.979 | 0.989 |
| `mask_rank` | 0.613 | 0.990 | 0.979 | 0.989 |

Predictor sets: **controls3** = log_history, small_margin, history_popularity; **controls** = log_history, small_margin, history_popularity, rec_popularity, top_margin, entropy_t0.1; **trajectory** = u_set_unif, u_rank_unif, score_change_unif, score_change_first, cons_set, cons_rank, reversal_set; **controls3 + trajectory** = log_history, small_margin, history_popularity, u_set_unif, u_rank_unif, score_change_unif, score_change_first, cons_set, cons_rank, reversal_set; **combined** = log_history, small_margin, history_popularity, rec_popularity, top_margin, entropy_t0.1, u_set_unif, u_rank_unif, score_change_unif, score_change_first, cons_set, cons_rank, reversal_set; **controls3 + lists** = log_history, small_margin, history_popularity, u_set_unif, u_rank_unif, cons_set, cons_rank, reversal_set; **controls + lists** = log_history, small_margin, history_popularity, rec_popularity, top_margin, entropy_t0.1, u_set_unif, u_rank_unif, cons_set, cons_rank, reversal_set; **controls3 + sampling2** = log_history, small_margin, history_popularity, samp2_set, samp2_rank, samp2_score; **controls3 + sampling5** = log_history, small_margin, history_popularity, samp5_set, samp5_rank, samp5_score; **controls3 + sampling10** = log_history, small_margin, history_popularity, samp10_set, samp10_rank, samp10_score; **controls + sampling2** = log_history, small_margin, history_popularity, rec_popularity, top_margin, entropy_t0.1, samp2_set, samp2_rank, samp2_score; **controls + sampling5** = log_history, small_margin, history_popularity, rec_popularity, top_margin, entropy_t0.1, samp5_set, samp5_rank, samp5_score; **controls + sampling10** = log_history, small_margin, history_popularity, rec_popularity, top_margin, entropy_t0.1, samp10_set, samp10_rank, samp10_score; **controls3 + masking** = log_history, small_margin, history_popularity, mask_set, mask_rank, mask_score, mask_unstable; **controls + masking** = log_history, small_margin, history_popularity, rec_popularity, top_margin, entropy_t0.1, mask_set, mask_rank, mask_score, mask_unstable; **controls + lists + sampling10** = log_history, small_margin, history_popularity, rec_popularity, top_margin, entropy_t0.1, u_set_unif, u_rank_unif, cons_set, cons_rank, reversal_set, samp10_set, samp10_rank, samp10_score; **controls + lists + masking** = log_history, small_margin, history_popularity, rec_popularity, top_margin, entropy_t0.1, u_set_unif, u_rank_unif, cons_set, cons_rank, reversal_set, mask_set, mask_rank, mask_score, mask_unstable; **all** = log_history, small_margin, history_popularity, rec_popularity, top_margin, entropy_t0.1, u_set_unif, u_rank_unif, score_change_unif, score_change_first, cons_set, cons_rank, reversal_set, samp10_set, samp10_rank, samp10_score, mask_set, mask_rank, mask_score, mask_unstable

Reliability tables of the combined judge: `reliability_<run>.csv`.
