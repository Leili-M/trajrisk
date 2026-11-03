# Section 7: calibration transfer to unseen users, amazon_video_games (DiffRec)

Risk model fitted on valid_risk, applied once to test. Seen: test users who also have risk-validation cases; unseen: all other test users. Share of test cases from unseen users: 0.847. Mean over runs with 95% t-interval.

| Predictors | Users | Cases | Miss rate | Mean predicted | Calibration gap | Brier | AUROC |
|---|---|---|---|---|---|---|---|
| `controls` | seen | 4,038 | 0.947 | 0.925 | -0.021 [-0.022, -0.020] | 0.051 [0.049, 0.053] | 0.603 [0.596, 0.611] |
| `controls` | unseen | 22,312 | 0.926 | 0.918 | -0.008 [-0.011, -0.006] | 0.067 [0.066, 0.068] | 0.645 [0.640, 0.650] |
| `controls + lists` | seen | 4,038 | 0.947 | 0.926 | -0.021 [-0.022, -0.019] | 0.051 [0.049, 0.053] | 0.618 [0.603, 0.634] |
| `controls + lists` | unseen | 22,312 | 0.926 | 0.918 | -0.008 [-0.011, -0.006] | 0.066 [0.065, 0.068] | 0.667 [0.663, 0.671] |
| `combined` | seen | 4,038 | 0.947 | 0.926 | -0.021 [-0.023, -0.020] | 0.051 [0.049, 0.053] | 0.617 [0.605, 0.630] |
| `combined` | unseen | 22,312 | 0.926 | 0.918 | -0.008 [-0.010, -0.006] | 0.066 [0.065, 0.068] | 0.667 [0.662, 0.671] |

Calibration gap: mean predicted miss probability minus observed miss rate (0 = calibrated in the large).
