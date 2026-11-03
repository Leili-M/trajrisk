# Section 7: calibration transfer to unseen users, base (DiffRec)

Risk model fitted on valid_risk, applied once to test. Seen: test users who also have risk-validation cases; unseen: all other test users. Share of test cases from unseen users: 0.408. Mean over runs with 95% t-interval.

| Predictors | Users | Cases | Miss rate | Mean predicted | Calibration gap | Brier | AUROC |
|---|---|---|---|---|---|---|---|
| `controls` | seen | 35,789 | 0.988 | 0.987 | -0.001 [-0.002, -0.001] | 0.012 [0.011, 0.012] | 0.706 [0.686, 0.726] |
| `controls` | unseen | 24,669 | 0.979 | 0.977 | -0.002 [-0.003, 0.000] | 0.021 [0.019, 0.022] | 0.679 [0.671, 0.687] |
| `controls + lists` | seen | 35,789 | 0.988 | 0.987 | -0.001 [-0.001, -0.001] | 0.012 [0.011, 0.012] | 0.704 [0.682, 0.726] |
| `controls + lists` | unseen | 24,669 | 0.979 | 0.978 | -0.001 [-0.003, 0.000] | 0.021 [0.019, 0.022] | 0.677 [0.666, 0.687] |
| `combined` | seen | 35,789 | 0.988 | 0.987 | -0.001 [-0.002, -0.001] | 0.012 [0.011, 0.012] | 0.704 [0.681, 0.726] |
| `combined` | unseen | 24,669 | 0.979 | 0.978 | -0.001 [-0.002, 0.001] | 0.021 [0.019, 0.022] | 0.673 [0.665, 0.682] |

Calibration gap: mean predicted miss probability minus observed miss rate (0 = calibrated in the large).
