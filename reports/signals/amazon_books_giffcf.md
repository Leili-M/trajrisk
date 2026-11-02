# Section 5 signals: amazon_books (GiffCF)

Single-signal AUROC on **valid_risk** (miss = target not in final top-10).
Higher signal = higher risk; AUROC < 0.5 means the signal points the other way.
Choices below are made on this split only; the test split is not used.

Runs: bk_giff_s1 (104,919 cases, miss 0.868), bk_giff_s2 (104,919 cases, miss 0.867), bk_giff_s3 (104,919 cases, miss 0.868)

| Group | Signal | Mean AUROC | SD | bk_giff_s1 | bk_giff_s2 | bk_giff_s3 |
|---|---|---|---|---|---|---|
| U_set | `u_set_unif` | 0.524 | 0.009 | 0.514 | 0.527 | 0.529 |
| U_set | `u_set_r0.75` | 0.523 | 0.008 | 0.514 | 0.527 | 0.529 |
| U_set | `u_set_r0.5` | 0.523 | 0.008 | 0.514 | 0.527 | 0.528 |
| U_set | `u_set_r0.25` | 0.523 | 0.008 | 0.514 | 0.527 | 0.527 |
| U_set | `u_set_early` | 0.524 | 0.011 | 0.512 | 0.528 | 0.533 |
| U_set | `u_set_late` | 0.519 | 0.009 | 0.509 | 0.527 | 0.520 |
| U_set | `u_set50_unif` | 0.470 | 0.018 | 0.458 | 0.490 | 0.461 |
| U_rank | `u_rank_unif` | 0.559 | 0.009 | 0.549 | 0.563 | 0.565 |
| U_rank | `u_rank_r0.75` | 0.558 | 0.009 | 0.548 | 0.563 | 0.563 |
| U_rank | `u_rank_r0.5` | 0.557 | 0.009 | 0.547 | 0.563 | 0.560 |
| U_rank | `u_rank_r0.25` | 0.555 | 0.009 | 0.546 | 0.563 | 0.557 |
| U_rank | `u_rank_early` | 0.561 | 0.010 | 0.550 | 0.563 | 0.570 |
| U_rank | `u_rank_late` | 0.554 | 0.010 | 0.544 | 0.563 | 0.556 |
| consecutive | `cons_set` | 0.526 | 0.007 | 0.518 | 0.527 | 0.533 |
| consecutive | `cons_rank` | 0.564 | 0.010 | 0.553 | 0.563 | 0.574 |
| consecutive | `reversal_set` | 0.508 | 0.007 | 0.511 | 0.500 | 0.512 |
| score change | `score_change_first` | 0.495 | 0.157 | 0.424 | 0.675 | 0.386 |
| score change | `score_change_unif` | 0.489 | 0.163 | 0.418 | 0.675 | 0.374 |
| score change | `score_change_r0.75` | 0.488 | 0.163 | 0.417 | 0.675 | 0.373 |
| score change | `score_change_r0.5` | 0.488 | 0.164 | 0.417 | 0.675 | 0.371 |
| score change | `score_change_r0.25` | 0.487 | 0.165 | 0.416 | 0.675 | 0.370 |
| score change | `score_change_early` | 0.495 | 0.157 | 0.424 | 0.675 | 0.386 |
| score change | `score_change_late` | 0.485 | 0.166 | 0.415 | 0.675 | 0.367 |
| final output | `small_margin` | 0.581 | 0.006 | 0.574 | 0.583 | 0.585 |
| final output | `top_margin` | 0.606 | 0.007 | 0.606 | 0.614 | 0.599 |
| final output | `entropy_t0.1` | 0.638 | 0.009 | 0.637 | 0.648 | 0.631 |
| final output | `entropy_t0.3` | 0.671 | 0.004 | 0.669 | 0.675 | 0.670 |
| final output | `entropy_t1` | 0.681 | 0.003 | 0.678 | 0.681 | 0.684 |
| final output | `entropy_t3` | 0.680 | 0.003 | 0.677 | 0.680 | 0.682 |
| controls | `log_history` | 0.500 | 0.002 | 0.498 | 0.503 | 0.500 |
| controls | `history_popularity` | 0.504 | 0.001 | 0.503 | 0.503 | 0.505 |
| controls | `rec_popularity` | 0.455 | 0.006 | 0.453 | 0.461 | 0.450 |

## Choices on this split

- u_set_weighting: `unif` (mean AUROC 0.524)
- u_rank_weighting: `unif` (mean AUROC 0.559)
- score_change_weighting: `unif` (mean AUROC 0.489)
- entropy_temperature: `t1` (mean AUROC 0.681)
