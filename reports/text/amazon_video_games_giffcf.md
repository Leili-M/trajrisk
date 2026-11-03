# Section 7: text (MiniLM) baselines, amazon_video_games (GiffCF)

Risk model fitted on valid_risk, applied once to test; mean over runs with 95% t-interval. Controls = all final-output and history controls of the main analysis.

## AUROC gains on test

| Comparison | Gain [95% CI] |
|---|---|
| Text (static) over controls | +0.022 [+0.012, +0.033] |
| Lists over controls + text | -0.001 [-0.004, +0.002] |
| Trajectory over controls + text | +0.001 [-0.007, +0.009] |
| Semantic drift over controls + lists (proposal: generic semantic list similarity) | +0.001 [-0.000, +0.001] |
| Text (static) over combined controls + lists | +0.021 [+0.013, +0.030] |

## AUROC and Brier of each predictor set (test)

| Predictors | AUROC | Brier |
|---|---|---|
| `controls` | 0.675 | 0.0605 |
| `controls + lists` | 0.675 | 0.0605 |
| `combined` | 0.679 | 0.0604 |
| `controls + text` | 0.697 | 0.0598 |
| `controls + text + lists` | 0.697 | 0.0599 |
| `controls + text + trajectory` | 0.699 | 0.0598 |
| `controls + lists + semantic drift` | 0.676 | 0.0604 |
| `controls + text + lists + semantic drift` | 0.696 | 0.0599 |

## Untrained single text signals (test AUROC; 0.5 = chance)

| Signal | AUROC |
|---|---|
| `list_spread` | 0.628 |
| `hist_to_list` | 0.598 |
| `hist_spread` | 0.513 |
| `traj_sem` | 0.520 |

Predictor sets: **controls** = log_history, small_margin, history_popularity, rec_popularity, top_margin, entropy_t1; **controls + lists** = log_history, small_margin, history_popularity, rec_popularity, top_margin, entropy_t1, u_set_unif, u_rank_r0.75, cons_set, cons_rank, reversal_set; **combined** = log_history, small_margin, history_popularity, rec_popularity, top_margin, entropy_t1, u_set_unif, u_rank_r0.75, score_change_unif, score_change_first, cons_set, cons_rank, reversal_set; **controls + text** = log_history, small_margin, history_popularity, rec_popularity, top_margin, entropy_t1, list_spread, hist_to_list, hist_spread; **controls + text + lists** = log_history, small_margin, history_popularity, rec_popularity, top_margin, entropy_t1, list_spread, hist_to_list, hist_spread, u_set_unif, u_rank_r0.75, cons_set, cons_rank, reversal_set; **controls + text + trajectory** = log_history, small_margin, history_popularity, rec_popularity, top_margin, entropy_t1, list_spread, hist_to_list, hist_spread, u_set_unif, u_rank_r0.75, score_change_unif, score_change_first, cons_set, cons_rank, reversal_set; **controls + lists + semantic drift** = log_history, small_margin, history_popularity, rec_popularity, top_margin, entropy_t1, u_set_unif, u_rank_r0.75, cons_set, cons_rank, reversal_set, traj_sem; **controls + text + lists + semantic drift** = log_history, small_margin, history_popularity, rec_popularity, top_margin, entropy_t1, list_spread, hist_to_list, hist_spread, u_set_unif, u_rank_r0.75, cons_set, cons_rank, reversal_set, traj_sem
