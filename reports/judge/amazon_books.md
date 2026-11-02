# Sections 7-8: judge on amazon_books (DiffRec)

Judge fitted on **valid_risk**, evaluated once on **test**. Mean over runs with 95% t-interval.
Runs: bk_seed42 (test 253,631 cases, miss 0.924), bk_seed1 (test 253,631 cases, miss 0.925), bk_seed2 (test 253,631 cases, miss 0.925), bk_seed3 (test 253,631 cases, miss 0.925), bk_seed4 (test 253,631 cases, miss 0.924)

## Primary endpoint: AUROC gain (success if >= 0.02)

| Comparison | Gain [95% CI] |
|---|---|
| combined over controls3 | 0.091 [0.086, 0.097] |
| controls3 + trajectory over controls3 | 0.065 [0.055, 0.075] |
| combined over controls | 0.011 [0.007, 0.015] |
| CV inside test (earlier protocol): controls3 + trajectory over controls3 | 0.067 [0.057, 0.077] |

## Predictor sets (test)

| Predictors | AUROC | AUPRC | Brier | AURC (miss) | AURC (1-NDCG) |
|---|---|---|---|---|---|
| controls3 | 0.659 [0.658, 0.661] | 0.959 [0.958, 0.959] | 0.069 [0.068, 0.069] | 0.889 [0.888, 0.890] | 0.936 [0.935, 0.936] |
| controls | 0.740 [0.738, 0.742] | 0.970 [0.970, 0.971] | 0.067 [0.067, 0.067] | 0.865 [0.864, 0.867] | 0.919 [0.918, 0.920] |
| trajectory | 0.699 [0.689, 0.708] | 0.964 [0.963, 0.966] | 0.067 [0.067, 0.068] | 0.873 [0.869, 0.877] | 0.926 [0.924, 0.929] |
| controls3 + trajectory | 0.725 [0.715, 0.734] | 0.968 [0.967, 0.969] | 0.066 [0.066, 0.067] | 0.866 [0.863, 0.869] | 0.920 [0.918, 0.922] |
| combined | 0.751 [0.746, 0.756] | 0.971 [0.971, 0.972] | 0.065 [0.065, 0.065] | 0.855 [0.853, 0.857] | 0.912 [0.911, 0.914] |

Failure prevalence (AUPRC of a random score): 0.925. References (not deployable): AURC miss random 0.924 / oracle 0.730; AURC 1-NDCG random 0.956 / oracle 0.825.

## Untrained single signals (test, mean over runs)

| Signal | AUROC | AUPRC | AURC (miss) | AURC (1-NDCG) |
|---|---|---|---|---|
| `u_set_unif` | 0.605 | 0.942 | 0.905 | 0.946 |
| `u_rank_unif` | 0.676 | 0.957 | 0.887 | 0.935 |
| `score_change_unif` | 0.637 | 0.957 | 0.899 | 0.942 |
| `score_change_first` | 0.640 | 0.957 | 0.896 | 0.940 |
| `cons_set` | 0.604 | 0.941 | 0.905 | 0.946 |
| `small_margin` | 0.619 | 0.946 | 0.892 | 0.940 |

Predictor sets: **controls3** = log_history, small_margin, history_popularity; **controls** = log_history, small_margin, history_popularity, rec_popularity, top_margin, entropy_t3; **trajectory** = u_set_unif, u_rank_unif, score_change_unif, score_change_first, cons_set, cons_rank, reversal_set; **controls3 + trajectory** = log_history, small_margin, history_popularity, u_set_unif, u_rank_unif, score_change_unif, score_change_first, cons_set, cons_rank, reversal_set; **combined** = log_history, small_margin, history_popularity, rec_popularity, top_margin, entropy_t3, u_set_unif, u_rank_unif, score_change_unif, score_change_first, cons_set, cons_rank, reversal_set

Reliability tables of the combined judge: `reliability_<run>.csv`.
