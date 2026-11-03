# Section 7: calibration transfer to unseen users, amazon_video_games (GiffCF)

Risk model fitted on valid_risk, applied once to test. Seen: test users who also have risk-validation cases; unseen: all other test users. Share of test cases from unseen users: 0.847. Mean over runs with 95% t-interval.

| Predictors | Users | Cases | Miss rate | Mean predicted | Calibration gap | Brier | AUROC |
|---|---|---|---|---|---|---|---|
| `controls` | seen | 4,038 | 0.942 | 0.923 | -0.019 [-0.027, -0.011] | 0.056 [0.048, 0.064] | 0.633 [0.591, 0.676] |
| `controls` | unseen | 22,312 | 0.932 | 0.921 | -0.011 [-0.015, -0.008] | 0.061 [0.052, 0.071] | 0.682 [0.611, 0.752] |
| `controls + lists` | seen | 4,038 | 0.942 | 0.923 | -0.019 [-0.029, -0.009] | 0.056 [0.047, 0.064] | 0.634 [0.592, 0.675] |
| `controls + lists` | unseen | 22,312 | 0.932 | 0.921 | -0.011 [-0.015, -0.008] | 0.061 [0.052, 0.071] | 0.682 [0.607, 0.757] |
| `combined` | seen | 4,038 | 0.942 | 0.922 | -0.020 [-0.032, -0.007] | 0.056 [0.047, 0.064] | 0.633 [0.600, 0.665] |
| `combined` | unseen | 22,312 | 0.932 | 0.921 | -0.011 [-0.014, -0.009] | 0.061 [0.052, 0.071] | 0.686 [0.607, 0.765] |

Calibration gap: mean predicted miss probability minus observed miss rate (0 = calibrated in the large).
