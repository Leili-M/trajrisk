# Section 5 signals: amazon_video_games (DiffRec)

Single-signal AUROC on **valid_risk** (miss = target not in final top-10).
Higher signal = higher risk; AUROC < 0.5 means the signal points the other way.
Choices below are made on this split only; the test split is not used.

Runs: vg_base (14,835 cases, miss 0.920), vg_seed1 (14,835 cases, miss 0.917), vg_seed2 (14,835 cases, miss 0.919), vg_seed3 (14,835 cases, miss 0.919), vg_seed4 (14,835 cases, miss 0.919)

| Group | Signal | Mean AUROC | SD | vg_base | vg_seed1 | vg_seed2 | vg_seed3 | vg_seed4 |
|---|---|---|---|---|---|---|---|---|
| U_set | `u_set_unif` | 0.562 | 0.010 | 0.557 | 0.560 | 0.564 | 0.578 | 0.551 |
| U_set | `u_set_r0.75` | 0.563 | 0.010 | 0.557 | 0.561 | 0.565 | 0.578 | 0.552 |
| U_set | `u_set_r0.5` | 0.562 | 0.010 | 0.557 | 0.561 | 0.565 | 0.577 | 0.552 |
| U_set | `u_set_r0.25` | 0.562 | 0.010 | 0.556 | 0.561 | 0.565 | 0.577 | 0.552 |
| U_set | `u_set_early` | 0.557 | 0.012 | 0.554 | 0.557 | 0.560 | 0.573 | 0.540 |
| U_set | `u_set_late` | 0.559 | 0.009 | 0.556 | 0.554 | 0.561 | 0.573 | 0.552 |
| U_set | `u_set50_unif` | 0.537 | 0.006 | 0.544 | 0.542 | 0.534 | 0.532 | 0.534 |
| U_rank | `u_rank_unif` | 0.598 | 0.009 | 0.607 | 0.589 | 0.587 | 0.607 | 0.600 |
| U_rank | `u_rank_r0.75` | 0.597 | 0.009 | 0.607 | 0.589 | 0.587 | 0.606 | 0.598 |
| U_rank | `u_rank_r0.5` | 0.596 | 0.009 | 0.607 | 0.587 | 0.587 | 0.605 | 0.594 |
| U_rank | `u_rank_r0.25` | 0.594 | 0.010 | 0.606 | 0.585 | 0.587 | 0.603 | 0.589 |
| U_rank | `u_rank_early` | 0.597 | 0.010 | 0.606 | 0.589 | 0.585 | 0.605 | 0.600 |
| U_rank | `u_rank_late` | 0.594 | 0.010 | 0.606 | 0.583 | 0.587 | 0.603 | 0.592 |
| consecutive | `cons_set` | 0.561 | 0.008 | 0.561 | 0.556 | 0.563 | 0.574 | 0.553 |
| consecutive | `cons_rank` | 0.597 | 0.008 | 0.606 | 0.592 | 0.586 | 0.603 | 0.596 |
| consecutive | `reversal_set` | 0.541 | 0.007 | 0.538 | 0.531 | 0.544 | 0.549 | 0.542 |
| score change | `score_change_first` | 0.535 | 0.003 | 0.538 | 0.538 | 0.533 | 0.530 | 0.534 |
| score change | `score_change_unif` | 0.534 | 0.003 | 0.537 | 0.537 | 0.533 | 0.530 | 0.533 |
| score change | `score_change_r0.75` | 0.534 | 0.003 | 0.537 | 0.537 | 0.533 | 0.530 | 0.533 |
| score change | `score_change_r0.5` | 0.534 | 0.003 | 0.537 | 0.537 | 0.533 | 0.530 | 0.533 |
| score change | `score_change_r0.25` | 0.534 | 0.003 | 0.537 | 0.537 | 0.533 | 0.530 | 0.533 |
| score change | `score_change_early` | 0.534 | 0.003 | 0.538 | 0.537 | 0.533 | 0.530 | 0.534 |
| score change | `score_change_late` | 0.534 | 0.003 | 0.537 | 0.537 | 0.533 | 0.530 | 0.533 |
| final output | `small_margin` | 0.546 | 0.007 | 0.542 | 0.558 | 0.543 | 0.540 | 0.548 |
| final output | `top_margin` | 0.572 | 0.007 | 0.583 | 0.571 | 0.564 | 0.569 | 0.573 |
| final output | `entropy_t0.1` | 0.586 | 0.007 | 0.598 | 0.584 | 0.579 | 0.583 | 0.587 |
| final output | `entropy_t0.3` | 0.616 | 0.007 | 0.627 | 0.615 | 0.613 | 0.612 | 0.611 |
| final output | `entropy_t1` | 0.626 | 0.006 | 0.636 | 0.626 | 0.623 | 0.621 | 0.623 |
| final output | `entropy_t3` | 0.625 | 0.005 | 0.633 | 0.625 | 0.622 | 0.620 | 0.623 |
| controls | `log_history` | 0.496 | 0.006 | 0.488 | 0.491 | 0.502 | 0.502 | 0.494 |
| controls | `history_popularity` | 0.448 | 0.008 | 0.456 | 0.449 | 0.436 | 0.444 | 0.454 |
| controls | `rec_popularity` | 0.560 | 0.010 | 0.564 | 0.548 | 0.563 | 0.572 | 0.552 |

## Choices on this split

- u_set_weighting: `r0.75` (mean AUROC 0.563)
- u_rank_weighting: `unif` (mean AUROC 0.598)
- score_change_weighting: `unif` (mean AUROC 0.534)
- entropy_temperature: `t1` (mean AUROC 0.626)
