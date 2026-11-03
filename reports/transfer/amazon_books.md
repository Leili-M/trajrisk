# Section 7: calibration transfer to unseen users, amazon_books (DiffRec)

Risk model fitted on valid_risk, applied once to test. Seen: test users who also have risk-validation cases; unseen: all other test users. Share of test cases from unseen users: 0.739. Mean over runs with 95% t-interval.

| Predictors | Users | Cases | Miss rate | Mean predicted | Calibration gap | Brier | AUROC |
|---|---|---|---|---|---|---|---|
| `controls` | seen | 66,108 | 0.927 | 0.920 | -0.007 [-0.008, -0.005] | 0.067 [0.067, 0.068] | 0.697 [0.692, 0.702] |
| `controls` | unseen | 187,523 | 0.924 | 0.921 | -0.002 [-0.003, -0.002] | 0.067 [0.066, 0.067] | 0.753 [0.751, 0.756] |
| `controls + lists` | seen | 66,108 | 0.927 | 0.920 | -0.007 [-0.008, -0.006] | 0.067 [0.066, 0.067] | 0.708 [0.706, 0.710] |
| `controls + lists` | unseen | 187,523 | 0.924 | 0.922 | -0.002 [-0.003, -0.002] | 0.066 [0.065, 0.066] | 0.762 [0.755, 0.768] |
| `combined` | seen | 66,108 | 0.927 | 0.922 | -0.005 [-0.007, -0.004] | 0.066 [0.066, 0.067] | 0.696 [0.691, 0.701] |
| `combined` | unseen | 187,523 | 0.924 | 0.922 | -0.002 [-0.003, -0.002] | 0.065 [0.064, 0.065] | 0.767 [0.762, 0.773] |

Calibration gap: mean predicted miss probability minus observed miss rate (0 = calibrated in the large).
