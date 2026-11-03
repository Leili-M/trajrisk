# Section 7: calibration transfer to unseen users, amazon_books (GiffCF)

Risk model fitted on valid_risk, applied once to test. Seen: test users who also have risk-validation cases; unseen: all other test users. Share of test cases from unseen users: 0.739. Mean over runs with 95% t-interval.

| Predictors | Users | Cases | Miss rate | Mean predicted | Calibration gap | Brier | AUROC |
|---|---|---|---|---|---|---|---|
| `controls` | seen | 66,108 | 0.890 | 0.880 | -0.010 [-0.011, -0.009] | 0.097 [0.096, 0.098] | 0.633 [0.610, 0.657] |
| `controls` | unseen | 187,523 | 0.869 | 0.866 | -0.003 [-0.004, -0.002] | 0.107 [0.104, 0.109] | 0.675 [0.642, 0.708] |
| `controls + lists` | seen | 66,108 | 0.890 | 0.880 | -0.009 [-0.011, -0.007] | 0.097 [0.097, 0.098] | 0.635 [0.618, 0.652] |
| `controls + lists` | unseen | 187,523 | 0.869 | 0.866 | -0.003 [-0.005, -0.002] | 0.106 [0.104, 0.109] | 0.683 [0.650, 0.716] |
| `combined` | seen | 66,108 | 0.890 | 0.881 | -0.009 [-0.016, -0.002] | 0.097 [0.094, 0.099] | 0.643 [0.628, 0.658] |
| `combined` | unseen | 187,523 | 0.869 | 0.865 | -0.004 [-0.005, -0.003] | 0.104 [0.102, 0.105] | 0.718 [0.695, 0.742] |

Calibration gap: mean predicted miss probability minus observed miss rate (0 = calibrated in the large).
