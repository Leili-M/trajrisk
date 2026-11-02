# Section 5 signals: amazon_video_games (GiffCF)

Single-signal AUROC on **valid_risk** (miss = target not in final top-10).
Higher signal = higher risk; AUROC < 0.5 means the signal points the other way.
Choices below are made on this split only; the test split is not used.

Runs: vg_giff_s1 (14,835 cases, miss 0.923), vg_giff_s2 (14,835 cases, miss 0.925), vg_giff_s3 (14,835 cases, miss 0.916)

| Group | Signal | Mean AUROC | SD | vg_giff_s1 | vg_giff_s2 | vg_giff_s3 |
|---|---|---|---|---|---|---|
| U_set | `u_set_unif` | 0.501 | 0.008 | 0.510 | 0.493 | 0.501 |
| U_set | `u_set_r0.75` | 0.501 | 0.008 | 0.510 | 0.493 | 0.501 |
| U_set | `u_set_r0.5` | 0.501 | 0.008 | 0.510 | 0.493 | 0.501 |
| U_set | `u_set_r0.25` | 0.501 | 0.008 | 0.510 | 0.493 | 0.501 |
| U_set | `u_set_early` | 0.502 | 0.009 | 0.512 | 0.495 | 0.498 |
| U_set | `u_set_late` | 0.501 | 0.007 | 0.508 | 0.494 | 0.503 |
| U_set | `u_set50_unif` | 0.481 | 0.009 | 0.473 | 0.480 | 0.490 |
| U_rank | `u_rank_unif` | 0.524 | 0.008 | 0.518 | 0.533 | 0.523 |
| U_rank | `u_rank_r0.75` | 0.525 | 0.007 | 0.518 | 0.532 | 0.523 |
| U_rank | `u_rank_r0.5` | 0.524 | 0.007 | 0.518 | 0.532 | 0.524 |
| U_rank | `u_rank_r0.25` | 0.524 | 0.007 | 0.518 | 0.531 | 0.524 |
| U_rank | `u_rank_early` | 0.525 | 0.010 | 0.517 | 0.536 | 0.521 |
| U_rank | `u_rank_late` | 0.524 | 0.006 | 0.518 | 0.531 | 0.524 |
| consecutive | `cons_set` | 0.502 | 0.008 | 0.507 | 0.492 | 0.506 |
| consecutive | `cons_rank` | 0.521 | 0.007 | 0.513 | 0.523 | 0.527 |
| consecutive | `reversal_set` | 0.499 | 0.004 | 0.498 | 0.495 | 0.503 |
| score change | `score_change_first` | 0.462 | 0.009 | 0.463 | 0.452 | 0.471 |
| score change | `score_change_unif` | 0.460 | 0.015 | 0.466 | 0.443 | 0.471 |
| score change | `score_change_r0.75` | 0.460 | 0.016 | 0.467 | 0.442 | 0.471 |
| score change | `score_change_r0.5` | 0.460 | 0.017 | 0.468 | 0.440 | 0.471 |
| score change | `score_change_r0.25` | 0.459 | 0.018 | 0.469 | 0.438 | 0.471 |
| score change | `score_change_early` | 0.462 | 0.009 | 0.463 | 0.452 | 0.471 |
| score change | `score_change_late` | 0.459 | 0.018 | 0.468 | 0.438 | 0.470 |
| final output | `small_margin` | 0.544 | 0.001 | 0.542 | 0.544 | 0.544 |
| final output | `top_margin` | 0.592 | 0.007 | 0.599 | 0.592 | 0.585 |
| final output | `entropy_t0.1` | 0.603 | 0.007 | 0.608 | 0.605 | 0.595 |
| final output | `entropy_t0.3` | 0.625 | 0.007 | 0.628 | 0.630 | 0.617 |
| final output | `entropy_t1` | 0.633 | 0.006 | 0.634 | 0.638 | 0.626 |
| final output | `entropy_t3` | 0.628 | 0.005 | 0.628 | 0.633 | 0.622 |
| controls | `log_history` | 0.468 | 0.007 | 0.474 | 0.470 | 0.460 |
| controls | `history_popularity` | 0.442 | 0.013 | 0.436 | 0.434 | 0.456 |
| controls | `rec_popularity` | 0.396 | 0.043 | 0.375 | 0.367 | 0.446 |

## Choices on this split

- u_set_weighting: `unif` (mean AUROC 0.501)
- u_rank_weighting: `r0.75` (mean AUROC 0.525)
- score_change_weighting: `unif` (mean AUROC 0.460)
- entropy_temperature: `t1` (mean AUROC 0.633)
