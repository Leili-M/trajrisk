# Section 5 signals: movielens (DiffRec)

Single-signal AUROC on **valid_risk** (miss = target not in final top-10).
Higher signal = higher risk; AUROC < 0.5 means the signal points the other way.
Choices below are made on this split only; the test split is not used.

Runs: ml_base (26,399 cases, miss 0.894), ml_seed1 (26,399 cases, miss 0.896), ml_seed2 (26,399 cases, miss 0.894), ml_seed3 (26,399 cases, miss 0.894), ml_seed4 (26,399 cases, miss 0.893)

| Group | Signal | Mean AUROC | SD | ml_base | ml_seed1 | ml_seed2 | ml_seed3 | ml_seed4 |
|---|---|---|---|---|---|---|---|---|
| U_set | `u_set_unif` | 0.456 | 0.004 | 0.455 | 0.458 | 0.461 | 0.455 | 0.452 |
| U_set | `u_set_r0.75` | 0.456 | 0.004 | 0.454 | 0.459 | 0.461 | 0.454 | 0.452 |
| U_set | `u_set_r0.5` | 0.456 | 0.004 | 0.454 | 0.460 | 0.461 | 0.454 | 0.452 |
| U_set | `u_set_r0.25` | 0.456 | 0.004 | 0.454 | 0.460 | 0.461 | 0.455 | 0.452 |
| U_set | `u_set_early` | 0.460 | 0.004 | 0.461 | 0.459 | 0.465 | 0.459 | 0.455 |
| U_set | `u_set_late` | 0.459 | 0.005 | 0.454 | 0.464 | 0.465 | 0.457 | 0.454 |
| U_set | `u_set50_unif` | 0.410 | 0.005 | 0.408 | 0.410 | 0.416 | 0.414 | 0.404 |
| U_rank | `u_rank_unif` | 0.432 | 0.006 | 0.426 | 0.431 | 0.442 | 0.430 | 0.432 |
| U_rank | `u_rank_r0.75` | 0.432 | 0.006 | 0.426 | 0.430 | 0.442 | 0.427 | 0.433 |
| U_rank | `u_rank_r0.5` | 0.432 | 0.007 | 0.426 | 0.430 | 0.442 | 0.426 | 0.435 |
| U_rank | `u_rank_r0.25` | 0.434 | 0.008 | 0.427 | 0.433 | 0.444 | 0.426 | 0.439 |
| U_rank | `u_rank_early` | 0.436 | 0.006 | 0.431 | 0.434 | 0.447 | 0.439 | 0.432 |
| U_rank | `u_rank_late` | 0.432 | 0.007 | 0.425 | 0.431 | 0.442 | 0.425 | 0.436 |
| consecutive | `cons_set` | 0.453 | 0.004 | 0.451 | 0.457 | 0.455 | 0.456 | 0.447 |
| consecutive | `cons_rank` | 0.431 | 0.006 | 0.426 | 0.427 | 0.442 | 0.429 | 0.433 |
| consecutive | `reversal_set` | 0.465 | 0.006 | 0.461 | 0.472 | 0.465 | 0.469 | 0.459 |
| score change | `score_change_first` | 0.389 | 0.004 | 0.389 | 0.394 | 0.392 | 0.388 | 0.385 |
| score change | `score_change_unif` | 0.389 | 0.004 | 0.388 | 0.394 | 0.392 | 0.387 | 0.384 |
| score change | `score_change_r0.75` | 0.389 | 0.004 | 0.388 | 0.393 | 0.392 | 0.387 | 0.384 |
| score change | `score_change_r0.5` | 0.389 | 0.004 | 0.388 | 0.393 | 0.392 | 0.387 | 0.384 |
| score change | `score_change_r0.25` | 0.389 | 0.004 | 0.388 | 0.393 | 0.392 | 0.387 | 0.384 |
| score change | `score_change_early` | 0.389 | 0.004 | 0.388 | 0.394 | 0.392 | 0.387 | 0.385 |
| score change | `score_change_late` | 0.389 | 0.004 | 0.388 | 0.393 | 0.392 | 0.387 | 0.384 |
| final output | `small_margin` | 0.463 | 0.004 | 0.462 | 0.460 | 0.470 | 0.461 | 0.462 |
| final output | `top_margin` | 0.473 | 0.005 | 0.468 | 0.468 | 0.480 | 0.474 | 0.473 |
| final output | `entropy_t0.1` | 0.458 | 0.006 | 0.453 | 0.456 | 0.467 | 0.456 | 0.455 |
| final output | `entropy_t0.3` | 0.422 | 0.004 | 0.421 | 0.423 | 0.426 | 0.420 | 0.416 |
| final output | `entropy_t1` | 0.407 | 0.003 | 0.408 | 0.410 | 0.411 | 0.405 | 0.402 |
| final output | `entropy_t3` | 0.404 | 0.003 | 0.403 | 0.406 | 0.407 | 0.402 | 0.399 |
| controls | `log_history` | 0.615 | 0.003 | 0.616 | 0.612 | 0.612 | 0.616 | 0.619 |
| controls | `history_popularity` | 0.408 | 0.002 | 0.408 | 0.410 | 0.410 | 0.409 | 0.404 |
| controls | `rec_popularity` | 0.441 | 0.003 | 0.442 | 0.445 | 0.440 | 0.440 | 0.439 |

## Choices on this split

- u_set_weighting: `r0.25` (mean AUROC 0.456)
- u_rank_weighting: `r0.25` (mean AUROC 0.434)
- score_change_weighting: `unif` (mean AUROC 0.389)
- entropy_temperature: `t0.1` (mean AUROC 0.458)
