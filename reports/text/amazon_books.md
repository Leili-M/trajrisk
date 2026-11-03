# Section 7: text (MiniLM) baselines, amazon_books (DiffRec)

Risk model fitted on valid_risk, applied once to test; mean over runs with 95% t-interval. Controls = all final-output and history controls of the main analysis.

## AUROC gains on test

| Comparison | Gain [95% CI] |
|---|---|
| Text (static) over controls | +0.033 [+0.032, +0.033] |
| Lists over controls + text | +0.002 [+0.000, +0.004] |
| Trajectory over controls + text | +0.002 [+0.000, +0.005] |
| Semantic drift over controls + lists (proposal: generic semantic list similarity) | +0.003 [+0.002, +0.005] |
| Text (static) over combined controls + lists | +0.026 [+0.023, +0.029] |

## AUROC and Brier of each predictor set (test)

| Predictors | AUROC | Brier |
|---|---|---|
| `controls` | 0.740 | 0.0667 |
| `controls + lists` | 0.749 | 0.0658 |
| `combined` | 0.751 | 0.0651 |
| `controls + text` | 0.773 | 0.0641 |
| `controls + text + lists` | 0.775 | 0.0640 |
| `controls + text + trajectory` | 0.775 | 0.0632 |
| `controls + lists + semantic drift` | 0.752 | 0.0658 |
| `controls + text + lists + semantic drift` | 0.775 | 0.0639 |

## Untrained single text signals (test AUROC; 0.5 = chance)

| Signal | AUROC |
|---|---|
| `list_spread` | 0.709 |
| `hist_to_list` | 0.674 |
| `hist_spread` | 0.542 |
| `traj_sem` | 0.612 |

Predictor sets: **controls** = log_history, small_margin, history_popularity, rec_popularity, top_margin, entropy_t3; **controls + lists** = log_history, small_margin, history_popularity, rec_popularity, top_margin, entropy_t3, u_set_unif, u_rank_unif, cons_set, cons_rank, reversal_set; **combined** = log_history, small_margin, history_popularity, rec_popularity, top_margin, entropy_t3, u_set_unif, u_rank_unif, score_change_unif, score_change_first, cons_set, cons_rank, reversal_set; **controls + text** = log_history, small_margin, history_popularity, rec_popularity, top_margin, entropy_t3, list_spread, hist_to_list, hist_spread; **controls + text + lists** = log_history, small_margin, history_popularity, rec_popularity, top_margin, entropy_t3, list_spread, hist_to_list, hist_spread, u_set_unif, u_rank_unif, cons_set, cons_rank, reversal_set; **controls + text + trajectory** = log_history, small_margin, history_popularity, rec_popularity, top_margin, entropy_t3, list_spread, hist_to_list, hist_spread, u_set_unif, u_rank_unif, score_change_unif, score_change_first, cons_set, cons_rank, reversal_set; **controls + lists + semantic drift** = log_history, small_margin, history_popularity, rec_popularity, top_margin, entropy_t3, u_set_unif, u_rank_unif, cons_set, cons_rank, reversal_set, traj_sem; **controls + text + lists + semantic drift** = log_history, small_margin, history_popularity, rec_popularity, top_margin, entropy_t3, list_spread, hist_to_list, hist_spread, u_set_unif, u_rank_unif, cons_set, cons_rank, reversal_set, traj_sem
