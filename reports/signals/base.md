# Section 5 signals: base (DiffRec)

Single-signal AUROC on **valid_risk** (miss = target not in final top-10).
Higher signal = higher risk; AUROC < 0.5 means the signal points the other way.
Choices below are made on this split only; the test split is not used.

Runs: hm_seed42 (23,891 cases, miss 0.984), hm_seed1 (23,891 cases, miss 0.985), hm_seed2 (23,891 cases, miss 0.984), hm_seed3 (23,891 cases, miss 0.984), hm_seed4 (23,891 cases, miss 0.982)

| Group | Signal | Mean AUROC | SD | hm_seed42 | hm_seed1 | hm_seed2 | hm_seed3 | hm_seed4 |
|---|---|---|---|---|---|---|---|---|
| U_set | `u_set_unif` | 0.467 | 0.023 | 0.480 | 0.498 | 0.438 | 0.462 | 0.457 |
| U_set | `u_set_r0.75` | 0.466 | 0.021 | 0.479 | 0.494 | 0.437 | 0.461 | 0.459 |
| U_set | `u_set_r0.5` | 0.466 | 0.020 | 0.478 | 0.492 | 0.438 | 0.461 | 0.459 |
| U_set | `u_set_r0.25` | 0.466 | 0.020 | 0.478 | 0.491 | 0.437 | 0.461 | 0.459 |
| U_set | `u_set_early` | 0.473 | 0.029 | 0.487 | 0.515 | 0.438 | 0.467 | 0.459 |
| U_set | `u_set_late` | 0.463 | 0.017 | 0.476 | 0.481 | 0.438 | 0.459 | 0.458 |
| U_set | `u_set50_unif` | 0.445 | 0.032 | 0.477 | 0.482 | 0.429 | 0.423 | 0.413 |
| U_rank | `u_rank_unif` | 0.453 | 0.029 | 0.463 | 0.498 | 0.436 | 0.441 | 0.424 |
| U_rank | `u_rank_r0.75` | 0.449 | 0.028 | 0.459 | 0.493 | 0.434 | 0.437 | 0.422 |
| U_rank | `u_rank_r0.5` | 0.445 | 0.026 | 0.455 | 0.486 | 0.433 | 0.434 | 0.419 |
| U_rank | `u_rank_r0.25` | 0.443 | 0.023 | 0.452 | 0.479 | 0.434 | 0.432 | 0.417 |
| U_rank | `u_rank_early` | 0.463 | 0.033 | 0.477 | 0.513 | 0.442 | 0.454 | 0.430 |
| U_rank | `u_rank_late` | 0.442 | 0.024 | 0.450 | 0.481 | 0.429 | 0.432 | 0.419 |
| consecutive | `cons_set` | 0.464 | 0.021 | 0.480 | 0.488 | 0.435 | 0.457 | 0.462 |
| consecutive | `cons_rank` | 0.445 | 0.028 | 0.450 | 0.493 | 0.426 | 0.435 | 0.424 |
| consecutive | `reversal_set` | 0.469 | 0.016 | 0.488 | 0.474 | 0.445 | 0.461 | 0.477 |
| score change | `score_change_first` | 0.368 | 0.019 | 0.390 | 0.379 | 0.373 | 0.354 | 0.344 |
| score change | `score_change_unif` | 0.363 | 0.017 | 0.383 | 0.375 | 0.365 | 0.352 | 0.341 |
| score change | `score_change_r0.75` | 0.362 | 0.017 | 0.382 | 0.374 | 0.364 | 0.351 | 0.341 |
| score change | `score_change_r0.5` | 0.362 | 0.016 | 0.381 | 0.373 | 0.363 | 0.351 | 0.340 |
| score change | `score_change_r0.25` | 0.361 | 0.016 | 0.380 | 0.372 | 0.362 | 0.351 | 0.340 |
| score change | `score_change_early` | 0.366 | 0.018 | 0.386 | 0.378 | 0.369 | 0.353 | 0.343 |
| score change | `score_change_late` | 0.361 | 0.016 | 0.380 | 0.372 | 0.362 | 0.351 | 0.340 |
| final output | `small_margin` | 0.449 | 0.020 | 0.465 | 0.468 | 0.444 | 0.417 | 0.451 |
| final output | `top_margin` | 0.476 | 0.019 | 0.459 | 0.504 | 0.462 | 0.468 | 0.484 |
| final output | `entropy_t0.1` | 0.445 | 0.015 | 0.448 | 0.469 | 0.436 | 0.436 | 0.433 |
| final output | `entropy_t0.3` | 0.434 | 0.015 | 0.437 | 0.458 | 0.431 | 0.422 | 0.421 |
| final output | `entropy_t1` | 0.430 | 0.014 | 0.434 | 0.453 | 0.429 | 0.419 | 0.417 |
| final output | `entropy_t3` | 0.429 | 0.014 | 0.434 | 0.451 | 0.428 | 0.419 | 0.416 |
| controls | `log_history` | 0.645 | 0.014 | 0.632 | 0.633 | 0.645 | 0.652 | 0.664 |
| controls | `history_popularity` | 0.303 | 0.020 | 0.300 | 0.271 | 0.318 | 0.322 | 0.304 |
| controls | `rec_popularity` | 0.329 | 0.011 | 0.340 | 0.317 | 0.334 | 0.338 | 0.318 |

## Choices on this split

- u_set_weighting: `unif` (mean AUROC 0.467)
- u_rank_weighting: `unif` (mean AUROC 0.453)
- score_change_weighting: `unif` (mean AUROC 0.363)
- entropy_temperature: `t0.1` (mean AUROC 0.445)
