# Section 7: calibration transfer to unseen users, movielens (DiffRec)

Risk model fitted on valid_risk, applied once to test. Seen: test users who also have risk-validation cases; unseen: all other test users. Share of test cases from unseen users: 0.848. Mean over runs with 95% t-interval.

| Predictors | Users | Cases | Miss rate | Mean predicted | Calibration gap | Brier | AUROC |
|---|---|---|---|---|---|---|---|
| `controls` | seen | 16,809 | 0.926 | 0.928 | 0.002 [0.001, 0.003] | 0.068 [0.067, 0.069] | 0.570 [0.566, 0.575] |
| `controls` | unseen | 93,648 | 0.902 | 0.902 | -0.000 [-0.002, 0.002] | 0.087 [0.087, 0.087] | 0.633 [0.630, 0.636] |
| `controls + lists` | seen | 16,809 | 0.926 | 0.929 | 0.003 [0.001, 0.004] | 0.068 [0.067, 0.069] | 0.570 [0.567, 0.574] |
| `controls + lists` | unseen | 93,648 | 0.902 | 0.902 | 0.000 [-0.002, 0.002] | 0.087 [0.087, 0.087] | 0.633 [0.631, 0.635] |
| `combined` | seen | 16,809 | 0.926 | 0.932 | 0.006 [0.004, 0.008] | 0.068 [0.067, 0.069] | 0.571 [0.567, 0.575] |
| `combined` | unseen | 93,648 | 0.902 | 0.902 | 0.001 [-0.001, 0.002] | 0.087 [0.086, 0.087] | 0.634 [0.632, 0.636] |

Calibration gap: mean predicted miss probability minus observed miss rate (0 = calibrated in the large).
