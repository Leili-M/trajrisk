# Section 7: text (MiniLM) baselines, amazon_video_games (DiffRec)

Risk model fitted on valid_risk, applied once to test; mean over runs with 95% t-interval. Controls = all final-output and history controls of the main analysis.

## AUROC gains on test

| Comparison | Gain [95% CI] |
|---|---|
| Text (static) over controls | +0.021 [+0.018, +0.024] |
| Lists over controls + text | +0.011 [+0.007, +0.015] |
| Trajectory over controls + text | +0.010 [+0.006, +0.015] |
| Semantic drift over controls + lists (proposal: generic semantic list similarity) | +0.000 [-0.000, +0.001] |
| Text (static) over combined controls + lists | +0.012 [+0.010, +0.015] |

## AUROC and Brier of each predictor set (test)

| Predictors | AUROC | Brier |
|---|---|---|
| `controls` | 0.642 | 0.0645 |
| `controls + lists` | 0.662 | 0.0640 |
| `combined` | 0.662 | 0.0640 |
| `controls + text` | 0.663 | 0.0641 |
| `controls + text + lists` | 0.675 | 0.0638 |
| `controls + text + trajectory` | 0.674 | 0.0638 |
| `controls + lists + semantic drift` | 0.663 | 0.0640 |
| `controls + text + lists + semantic drift` | 0.675 | 0.0638 |

## Untrained single text signals (test AUROC; 0.5 = chance)

| Signal | AUROC |
|---|---|
| `list_spread` | 0.611 |
| `hist_to_list` | 0.572 |
| `hist_spread` | 0.538 |
| `traj_sem` | 0.568 |

Predictor sets: **controls** = log_history, small_margin, history_popularity, rec_popularity, top_margin, entropy_t1; **controls + lists** = log_history, small_margin, history_popularity, rec_popularity, top_margin, entropy_t1, u_set_r0.75, u_rank_unif, cons_set, cons_rank, reversal_set; **combined** = log_history, small_margin, history_popularity, rec_popularity, top_margin, entropy_t1, u_set_r0.75, u_rank_unif, score_change_unif, score_change_first, cons_set, cons_rank, reversal_set; **controls + text** = log_history, small_margin, history_popularity, rec_popularity, top_margin, entropy_t1, list_spread, hist_to_list, hist_spread; **controls + text + lists** = log_history, small_margin, history_popularity, rec_popularity, top_margin, entropy_t1, list_spread, hist_to_list, hist_spread, u_set_r0.75, u_rank_unif, cons_set, cons_rank, reversal_set; **controls + text + trajectory** = log_history, small_margin, history_popularity, rec_popularity, top_margin, entropy_t1, list_spread, hist_to_list, hist_spread, u_set_r0.75, u_rank_unif, score_change_unif, score_change_first, cons_set, cons_rank, reversal_set; **controls + lists + semantic drift** = log_history, small_margin, history_popularity, rec_popularity, top_margin, entropy_t1, u_set_r0.75, u_rank_unif, cons_set, cons_rank, reversal_set, traj_sem; **controls + text + lists + semantic drift** = log_history, small_margin, history_popularity, rec_popularity, top_margin, entropy_t1, list_spread, hist_to_list, hist_spread, u_set_r0.75, u_rank_unif, cons_set, cons_rank, reversal_set, traj_sem
