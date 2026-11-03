# Section 7: text (MiniLM) baselines, amazon_books (GiffCF)

Risk model fitted on valid_risk, applied once to test; mean over runs with 95% t-interval. Controls = all final-output and history controls of the main analysis.

## AUROC gains on test

| Comparison | Gain [95% CI] |
|---|---|
| Text (static) over controls | +0.049 [+0.030, +0.069] |
| Lists over controls + text | +0.000 [-0.001, +0.002] |
| Trajectory over controls + text | +0.008 [-0.010, +0.027] |
| Semantic drift over controls + lists (proposal: generic semantic list similarity) | +0.007 [-0.003, +0.016] |
| Text (static) over combined controls + lists | +0.043 [+0.025, +0.061] |

## AUROC and Brier of each predictor set (test)

| Predictors | AUROC | Brier |
|---|---|---|
| `controls` | 0.665 | 0.1042 |
| `controls + lists` | 0.671 | 0.1039 |
| `combined` | 0.700 | 0.1018 |
| `controls + text` | 0.714 | 0.1014 |
| `controls + text + lists` | 0.714 | 0.1012 |
| `controls + text + trajectory` | 0.722 | 0.1003 |
| `controls + lists + semantic drift` | 0.678 | 0.1037 |
| `controls + text + lists + semantic drift` | 0.714 | 0.1012 |

## Untrained single text signals (test AUROC; 0.5 = chance)

| Signal | AUROC |
|---|---|
| `list_spread` | 0.652 |
| `hist_to_list` | 0.616 |
| `hist_spread` | 0.556 |
| `traj_sem` | 0.537 |

Predictor sets: **controls** = log_history, small_margin, history_popularity, rec_popularity, top_margin, entropy_t1; **controls + lists** = log_history, small_margin, history_popularity, rec_popularity, top_margin, entropy_t1, u_set_unif, u_rank_unif, cons_set, cons_rank, reversal_set; **combined** = log_history, small_margin, history_popularity, rec_popularity, top_margin, entropy_t1, u_set_unif, u_rank_unif, score_change_unif, score_change_first, cons_set, cons_rank, reversal_set; **controls + text** = log_history, small_margin, history_popularity, rec_popularity, top_margin, entropy_t1, list_spread, hist_to_list, hist_spread; **controls + text + lists** = log_history, small_margin, history_popularity, rec_popularity, top_margin, entropy_t1, list_spread, hist_to_list, hist_spread, u_set_unif, u_rank_unif, cons_set, cons_rank, reversal_set; **controls + text + trajectory** = log_history, small_margin, history_popularity, rec_popularity, top_margin, entropy_t1, list_spread, hist_to_list, hist_spread, u_set_unif, u_rank_unif, score_change_unif, score_change_first, cons_set, cons_rank, reversal_set; **controls + lists + semantic drift** = log_history, small_margin, history_popularity, rec_popularity, top_margin, entropy_t1, u_set_unif, u_rank_unif, cons_set, cons_rank, reversal_set, traj_sem; **controls + text + lists + semantic drift** = log_history, small_margin, history_popularity, rec_popularity, top_margin, entropy_t1, list_spread, hist_to_list, hist_spread, u_set_unif, u_rank_unif, cons_set, cons_rank, reversal_set, traj_sem
