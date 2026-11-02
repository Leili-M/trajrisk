# Section 5 signals: amazon_books (DiffRec)

Single-signal AUROC on **valid_risk** (miss = target not in final top-10).
Higher signal = higher risk; AUROC < 0.5 means the signal points the other way.
Choices below are made on this split only; the test split is not used.

Runs: bk_seed42 (104,919 cases, miss 0.919), bk_seed1 (104,919 cases, miss 0.920), bk_seed2 (104,919 cases, miss 0.920), bk_seed3 (104,919 cases, miss 0.920), bk_seed4 (104,919 cases, miss 0.920)

| Group | Signal | Mean AUROC | SD | bk_seed42 | bk_seed1 | bk_seed2 | bk_seed3 | bk_seed4 |
|---|---|---|---|---|---|---|---|---|
| U_set | `u_set_unif` | 0.594 | 0.008 | 0.598 | 0.599 | 0.601 | 0.582 | 0.592 |
| U_set | `u_set_r0.75` | 0.594 | 0.008 | 0.597 | 0.599 | 0.601 | 0.582 | 0.592 |
| U_set | `u_set_r0.5` | 0.594 | 0.008 | 0.597 | 0.599 | 0.601 | 0.582 | 0.591 |
| U_set | `u_set_r0.25` | 0.594 | 0.008 | 0.597 | 0.599 | 0.601 | 0.582 | 0.591 |
| U_set | `u_set_early` | 0.587 | 0.006 | 0.591 | 0.591 | 0.593 | 0.577 | 0.585 |
| U_set | `u_set_late` | 0.585 | 0.008 | 0.588 | 0.588 | 0.594 | 0.573 | 0.582 |
| U_set | `u_set50_unif` | 0.606 | 0.004 | 0.603 | 0.612 | 0.605 | 0.605 | 0.603 |
| U_rank | `u_rank_unif` | 0.662 | 0.009 | 0.670 | 0.672 | 0.663 | 0.650 | 0.655 |
| U_rank | `u_rank_r0.75` | 0.662 | 0.009 | 0.669 | 0.672 | 0.663 | 0.650 | 0.655 |
| U_rank | `u_rank_r0.5` | 0.661 | 0.009 | 0.667 | 0.671 | 0.662 | 0.649 | 0.653 |
| U_rank | `u_rank_r0.25` | 0.659 | 0.009 | 0.665 | 0.669 | 0.661 | 0.648 | 0.652 |
| U_rank | `u_rank_early` | 0.658 | 0.009 | 0.666 | 0.667 | 0.660 | 0.647 | 0.652 |
| U_rank | `u_rank_late` | 0.656 | 0.009 | 0.664 | 0.666 | 0.659 | 0.645 | 0.648 |
| consecutive | `cons_set` | 0.594 | 0.007 | 0.596 | 0.598 | 0.601 | 0.582 | 0.590 |
| consecutive | `cons_rank` | 0.662 | 0.009 | 0.668 | 0.671 | 0.664 | 0.650 | 0.655 |
| consecutive | `reversal_set` | 0.556 | 0.009 | 0.559 | 0.559 | 0.565 | 0.540 | 0.556 |
| score change | `score_change_first` | 0.636 | 0.002 | 0.636 | 0.639 | 0.634 | 0.637 | 0.634 |
| score change | `score_change_unif` | 0.633 | 0.002 | 0.633 | 0.636 | 0.631 | 0.633 | 0.630 |
| score change | `score_change_r0.75` | 0.632 | 0.002 | 0.632 | 0.636 | 0.631 | 0.633 | 0.629 |
| score change | `score_change_r0.5` | 0.632 | 0.002 | 0.631 | 0.635 | 0.630 | 0.632 | 0.629 |
| score change | `score_change_r0.25` | 0.631 | 0.002 | 0.631 | 0.635 | 0.630 | 0.632 | 0.628 |
| score change | `score_change_early` | 0.635 | 0.002 | 0.634 | 0.638 | 0.633 | 0.635 | 0.632 |
| score change | `score_change_late` | 0.631 | 0.002 | 0.631 | 0.635 | 0.630 | 0.632 | 0.628 |
| final output | `small_margin` | 0.606 | 0.007 | 0.609 | 0.615 | 0.601 | 0.597 | 0.607 |
| final output | `top_margin` | 0.630 | 0.003 | 0.631 | 0.634 | 0.627 | 0.627 | 0.628 |
| final output | `entropy_t0.1` | 0.670 | 0.002 | 0.670 | 0.675 | 0.668 | 0.670 | 0.669 |
| final output | `entropy_t0.3` | 0.702 | 0.002 | 0.701 | 0.707 | 0.701 | 0.702 | 0.701 |
| final output | `entropy_t1` | 0.720 | 0.002 | 0.718 | 0.723 | 0.720 | 0.718 | 0.718 |
| final output | `entropy_t3` | 0.721 | 0.002 | 0.720 | 0.724 | 0.721 | 0.719 | 0.719 |
| controls | `log_history` | 0.464 | 0.002 | 0.462 | 0.462 | 0.467 | 0.465 | 0.466 |
| controls | `history_popularity` | 0.380 | 0.002 | 0.382 | 0.378 | 0.379 | 0.379 | 0.382 |
| controls | `rec_popularity` | 0.657 | 0.009 | 0.673 | 0.656 | 0.651 | 0.651 | 0.655 |

## Choices on this split

- u_set_weighting: `unif` (mean AUROC 0.594)
- u_rank_weighting: `unif` (mean AUROC 0.662)
- score_change_weighting: `unif` (mean AUROC 0.633)
- entropy_temperature: `t3` (mean AUROC 0.721)
