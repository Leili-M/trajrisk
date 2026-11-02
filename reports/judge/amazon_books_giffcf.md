# Sections 7-8: judge on amazon_books (GiffCF)

Judge fitted on **valid_risk**, evaluated once on **test**. Mean over runs with 95% t-interval.
Runs: bk_giff_s1 (test 253,631 cases, miss 0.875), bk_giff_s2 (test 253,631 cases, miss 0.874), bk_giff_s3 (test 253,631 cases, miss 0.874)

## Primary endpoint: AUROC gain (success if >= 0.02)

| Comparison | Gain [95% CI] |
|---|---|
| combined over controls3 | 0.131 [0.109, 0.153] |
| controls3 + trajectory over controls3 | 0.098 [0.003, 0.193] |
| combined over controls | 0.036 [-0.010, 0.082] |
| controls3 + lists over controls3 | 0.025 [0.006, 0.043] |
| controls + lists over controls | 0.006 [0.003, 0.010] |
| CV inside test (earlier protocol): controls3 + trajectory over controls3 | 0.092 [-0.002, 0.187] |

## Predictor sets (test)

| Predictors | AUROC | AUPRC | Brier | AURC (miss) | AURC (1-NDCG) |
|---|---|---|---|---|---|
| controls3 | 0.569 [0.563, 0.575] | 0.897 [0.896, 0.898] | 0.109 [0.108, 0.109] | 0.838 [0.835, 0.842] | 0.901 [0.899, 0.903] |
| controls | 0.665 [0.635, 0.695] | 0.923 [0.915, 0.931] | 0.104 [0.102, 0.106] | 0.798 [0.786, 0.810] | 0.871 [0.863, 0.878] |
| trajectory | 0.653 [0.543, 0.762] | 0.920 [0.888, 0.951] | 0.105 [0.101, 0.110] | 0.807 [0.765, 0.848] | 0.880 [0.856, 0.904] |
| controls3 + trajectory | 0.667 [0.571, 0.763] | 0.923 [0.898, 0.948] | 0.104 [0.098, 0.111] | 0.798 [0.753, 0.843] | 0.874 [0.840, 0.908] |
| combined | 0.700 [0.677, 0.723] | 0.933 [0.926, 0.940] | 0.102 [0.100, 0.103] | 0.783 [0.772, 0.795] | 0.862 [0.855, 0.869] |
| controls3 + lists | 0.594 [0.569, 0.618] | 0.904 [0.899, 0.910] | 0.108 [0.108, 0.109] | 0.832 [0.822, 0.843] | 0.898 [0.893, 0.903] |
| controls + lists | 0.671 [0.643, 0.699] | 0.924 [0.917, 0.932] | 0.104 [0.102, 0.106] | 0.796 [0.784, 0.808] | 0.870 [0.862, 0.877] |

Failure prevalence (AUPRC of a random score): 0.874. References (not deployable): AURC miss random 0.875 / oracle 0.614; AURC 1-NDCG random 0.922 / oracle 0.733.

## Untrained single signals (test, mean over runs)

| Signal | AUROC | AUPRC | AURC (miss) | AURC (1-NDCG) |
|---|---|---|---|---|
| `u_set_unif` | 0.524 | 0.879 | 0.866 | 0.919 |
| `u_rank_unif` | 0.557 | 0.891 | 0.855 | 0.911 |
| `score_change_unif` | 0.488 | 0.865 | 0.873 | 0.921 |
| `score_change_first` | 0.495 | 0.870 | 0.871 | 0.920 |
| `cons_set` | 0.526 | 0.881 | 0.866 | 0.918 |
| `small_margin` | 0.589 | 0.901 | 0.837 | 0.905 |

Predictor sets: **controls3** = log_history, small_margin, history_popularity; **controls** = log_history, small_margin, history_popularity, rec_popularity, top_margin, entropy_t1; **trajectory** = u_set_unif, u_rank_unif, score_change_unif, score_change_first, cons_set, cons_rank, reversal_set; **controls3 + trajectory** = log_history, small_margin, history_popularity, u_set_unif, u_rank_unif, score_change_unif, score_change_first, cons_set, cons_rank, reversal_set; **combined** = log_history, small_margin, history_popularity, rec_popularity, top_margin, entropy_t1, u_set_unif, u_rank_unif, score_change_unif, score_change_first, cons_set, cons_rank, reversal_set; **controls3 + lists** = log_history, small_margin, history_popularity, u_set_unif, u_rank_unif, cons_set, cons_rank, reversal_set; **controls + lists** = log_history, small_margin, history_popularity, rec_popularity, top_margin, entropy_t1, u_set_unif, u_rank_unif, cons_set, cons_rank, reversal_set

Reliability tables of the combined judge: `reliability_<run>.csv`.
