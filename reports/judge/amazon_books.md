# Sections 7-8: judge on amazon_books (DiffRec)

Judge fitted on **valid_risk**, evaluated once on **test**. Mean over runs with 95% t-interval.
Runs: bk_seed42 (test 253,631 cases, miss 0.924), bk_seed1 (test 253,631 cases, miss 0.925), bk_seed2 (test 253,631 cases, miss 0.925), bk_seed3 (test 253,631 cases, miss 0.925), bk_seed4 (test 253,631 cases, miss 0.924)

## Primary endpoint: AUROC gain (success if >= 0.02)

| Comparison | Gain [95% CI] |
|---|---|
| combined over controls3 | 0.091 [0.086, 0.097] |
| controls3 + trajectory over controls3 | 0.065 [0.055, 0.075] |
| combined over controls | 0.011 [0.007, 0.015] |
| controls3 + lists over controls3 | 0.053 [0.042, 0.064] |
| controls + lists over controls | 0.009 [0.005, 0.013] |
| controls3 + sampling2 over controls3 | 0.081 [0.073, 0.090] |
| controls3 + sampling5 over controls3 | 0.086 [0.078, 0.095] |
| controls3 + sampling10 over controls3 | 0.088 [0.080, 0.097] |
| controls + sampling2 over controls | 0.021 [0.018, 0.024] |
| controls + sampling5 over controls | 0.024 [0.020, 0.027] |
| controls + sampling10 over controls | 0.025 [0.021, 0.028] |
| controls3 + masking over controls3 | 0.045 [0.041, 0.049] |
| controls + masking over controls | 0.006 [0.004, 0.008] |
| controls + lists + sampling10 over controls + lists | 0.016 [0.014, 0.017] |
| controls + lists + masking over controls + lists | 0.005 [0.004, 0.006] |
| all over controls | 0.028 [0.025, 0.031] |
| CV inside test (earlier protocol): controls3 + trajectory over controls3 | 0.067 [0.057, 0.077] |

## Predictor sets (test)

| Predictors | AUROC | AUPRC | Brier | AURC (miss) | AURC (1-NDCG) |
|---|---|---|---|---|---|
| controls3 | 0.659 [0.658, 0.661] | 0.959 [0.958, 0.959] | 0.069 [0.068, 0.069] | 0.889 [0.888, 0.890] | 0.936 [0.935, 0.936] |
| controls | 0.740 [0.738, 0.742] | 0.970 [0.970, 0.971] | 0.067 [0.067, 0.067] | 0.865 [0.864, 0.867] | 0.919 [0.918, 0.920] |
| trajectory | 0.699 [0.689, 0.708] | 0.964 [0.963, 0.966] | 0.067 [0.067, 0.068] | 0.873 [0.869, 0.877] | 0.926 [0.924, 0.929] |
| controls3 + trajectory | 0.725 [0.715, 0.734] | 0.968 [0.967, 0.969] | 0.066 [0.066, 0.067] | 0.866 [0.863, 0.869] | 0.920 [0.918, 0.922] |
| combined | 0.751 [0.746, 0.756] | 0.971 [0.971, 0.972] | 0.065 [0.065, 0.065] | 0.855 [0.853, 0.857] | 0.912 [0.911, 0.914] |
| controls3 + lists | 0.713 [0.702, 0.723] | 0.965 [0.963, 0.967] | 0.067 [0.066, 0.067] | 0.873 [0.870, 0.876] | 0.926 [0.924, 0.928] |
| controls + lists | 0.749 [0.744, 0.754] | 0.971 [0.970, 0.971] | 0.066 [0.065, 0.066] | 0.860 [0.857, 0.863] | 0.916 [0.914, 0.918] |
| controls3 + sampling2 | 0.741 [0.732, 0.749] | 0.970 [0.969, 0.971] | 0.066 [0.065, 0.066] | 0.860 [0.858, 0.863] | 0.917 [0.916, 0.919] |
| controls3 + sampling5 | 0.746 [0.737, 0.754] | 0.970 [0.969, 0.971] | 0.065 [0.065, 0.066] | 0.858 [0.856, 0.861] | 0.916 [0.915, 0.917] |
| controls3 + sampling10 | 0.748 [0.739, 0.756] | 0.971 [0.970, 0.972] | 0.065 [0.065, 0.066] | 0.857 [0.855, 0.860] | 0.915 [0.914, 0.917] |
| controls + sampling2 | 0.761 [0.757, 0.765] | 0.972 [0.972, 0.973] | 0.064 [0.064, 0.065] | 0.851 [0.849, 0.853] | 0.910 [0.909, 0.912] |
| controls + sampling5 | 0.764 [0.759, 0.768] | 0.973 [0.972, 0.973] | 0.064 [0.064, 0.065] | 0.850 [0.848, 0.851] | 0.910 [0.909, 0.911] |
| controls + sampling10 | 0.765 [0.761, 0.769] | 0.973 [0.972, 0.973] | 0.064 [0.064, 0.065] | 0.849 [0.848, 0.851] | 0.909 [0.908, 0.910] |
| controls3 + masking | 0.704 [0.700, 0.708] | 0.961 [0.960, 0.962] | 0.067 [0.066, 0.067] | 0.868 [0.866, 0.869] | 0.923 [0.922, 0.924] |
| controls + masking | 0.746 [0.743, 0.749] | 0.970 [0.969, 0.971] | 0.065 [0.065, 0.066] | 0.857 [0.856, 0.859] | 0.915 [0.914, 0.916] |
| controls + lists + sampling10 | 0.765 [0.761, 0.769] | 0.973 [0.972, 0.973] | 0.064 [0.064, 0.064] | 0.849 [0.848, 0.850] | 0.909 [0.908, 0.910] |
| controls + lists + masking | 0.754 [0.749, 0.759] | 0.971 [0.970, 0.972] | 0.065 [0.064, 0.065] | 0.854 [0.851, 0.857] | 0.913 [0.911, 0.915] |
| all | 0.768 [0.764, 0.773] | 0.973 [0.972, 0.974] | 0.063 [0.063, 0.064] | 0.845 [0.844, 0.847] | 0.907 [0.906, 0.908] |

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
| `samp10_set` | 0.680 | 0.958 | 0.885 | 0.935 |
| `samp10_rank` | 0.709 | 0.965 | 0.875 | 0.929 |
| `samp10_score` | 0.634 | 0.957 | 0.902 | 0.944 |
| `mask_set` | 0.569 | 0.942 | 0.918 | 0.952 |
| `mask_rank` | 0.566 | 0.943 | 0.919 | 0.952 |

Predictor sets: **controls3** = log_history, small_margin, history_popularity; **controls** = log_history, small_margin, history_popularity, rec_popularity, top_margin, entropy_t3; **trajectory** = u_set_unif, u_rank_unif, score_change_unif, score_change_first, cons_set, cons_rank, reversal_set; **controls3 + trajectory** = log_history, small_margin, history_popularity, u_set_unif, u_rank_unif, score_change_unif, score_change_first, cons_set, cons_rank, reversal_set; **combined** = log_history, small_margin, history_popularity, rec_popularity, top_margin, entropy_t3, u_set_unif, u_rank_unif, score_change_unif, score_change_first, cons_set, cons_rank, reversal_set; **controls3 + lists** = log_history, small_margin, history_popularity, u_set_unif, u_rank_unif, cons_set, cons_rank, reversal_set; **controls + lists** = log_history, small_margin, history_popularity, rec_popularity, top_margin, entropy_t3, u_set_unif, u_rank_unif, cons_set, cons_rank, reversal_set; **controls3 + sampling2** = log_history, small_margin, history_popularity, samp2_set, samp2_rank, samp2_score; **controls3 + sampling5** = log_history, small_margin, history_popularity, samp5_set, samp5_rank, samp5_score; **controls3 + sampling10** = log_history, small_margin, history_popularity, samp10_set, samp10_rank, samp10_score; **controls + sampling2** = log_history, small_margin, history_popularity, rec_popularity, top_margin, entropy_t3, samp2_set, samp2_rank, samp2_score; **controls + sampling5** = log_history, small_margin, history_popularity, rec_popularity, top_margin, entropy_t3, samp5_set, samp5_rank, samp5_score; **controls + sampling10** = log_history, small_margin, history_popularity, rec_popularity, top_margin, entropy_t3, samp10_set, samp10_rank, samp10_score; **controls3 + masking** = log_history, small_margin, history_popularity, mask_set, mask_rank, mask_score, mask_unstable; **controls + masking** = log_history, small_margin, history_popularity, rec_popularity, top_margin, entropy_t3, mask_set, mask_rank, mask_score, mask_unstable; **controls + lists + sampling10** = log_history, small_margin, history_popularity, rec_popularity, top_margin, entropy_t3, u_set_unif, u_rank_unif, cons_set, cons_rank, reversal_set, samp10_set, samp10_rank, samp10_score; **controls + lists + masking** = log_history, small_margin, history_popularity, rec_popularity, top_margin, entropy_t3, u_set_unif, u_rank_unif, cons_set, cons_rank, reversal_set, mask_set, mask_rank, mask_score, mask_unstable; **all** = log_history, small_margin, history_popularity, rec_popularity, top_margin, entropy_t3, u_set_unif, u_rank_unif, score_change_unif, score_change_first, cons_set, cons_rank, reversal_set, samp10_set, samp10_rank, samp10_score, mask_set, mask_rank, mask_score, mask_unstable

Reliability tables of the combined judge: `reliability_<run>.csv`.
