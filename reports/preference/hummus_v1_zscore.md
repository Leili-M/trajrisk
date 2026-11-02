# Section 6: preference-aware comparison (HUMMUS benchmark, DiffRec)

Runs: hm_seed42, hm_seed1, hm_seed2, hm_seed3, hm_seed4. Reranker (formula 9) chosen on validation scenarios: beta = 4, gamma = 10. Failure = regret >= 0.5. Detection rows: scenarios with feasible recipes; thresholds and judge from validation.

## Reranker grid (validation, soft pipeline)

| beta | gamma | valid mean regret |
|---|---|---|
| 0.000 | 0.000 | 0.995 |
| 0.000 | 1.000 | 0.986 |
| 0.000 | 10.000 | 0.956 |
| 0.250 | 0.000 | 0.995 |
| 0.250 | 1.000 | 0.985 |
| 0.250 | 10.000 | 0.955 |
| 0.500 | 0.000 | 0.995 |
| 0.500 | 1.000 | 0.984 |
| 0.500 | 10.000 | 0.954 |
| 1.000 | 0.000 | 0.994 |
| 1.000 | 1.000 | 0.983 |
| 1.000 | 10.000 | 0.953 |
| 2.000 | 0.000 | 0.992 |
| 2.000 | 1.000 | 0.980 |
| 2.000 | 10.000 | 0.950 |
| 4.000 | 0.000 | 0.989 |
| 4.000 | 1.000 | 0.976 |
| 4.000 | 10.000 | 0.947 |

## Outcomes on test

| pipeline | mean regret (many) | mean regret (few) | failure rate | lists with a violation | abstained when nothing feasible | regret, held-out combinations | regret, other | diversity |
|---|---|---|---|---|---|---|---|---|
| hard | 0.199 | 0.094 | 0.008 | 0.000 | 1.000 | 0.160 | 0.142 | 0.229 |
| plain | 0.990 | 1.000 | 1.000 | 1.000 | 0.000 | 0.995 | 0.995 | 0.298 |
| soft | 0.916 | 0.992 | 0.994 | 0.217 | 0.000 | 0.959 | 0.952 | 0.266 |

## Pipeline: soft

Test detection rows: 1800 (1789 failures).

### Single signals (test)

| signal | auroc | auroc_mean_over_runs | recall at matched threshold | false alarms among suitable |
|---|---|---|---|---|
| u_set | 0.322 | 0.285 | 1.000 | 1.000 |
| u_rank | 0.337 | 0.279 | 0.879 | 1.000 |
| u_attr | 0.292 | 0.253 | 1.000 | 1.000 |
| u_util | 0.387 | 0.391 | 1.000 | 1.000 |
| u_attr_rank | 0.323 | 0.293 | 0.911 | 1.000 |
| u_dist | 0.341 | 0.311 | 0.869 | 1.000 |
| u_generic | 0.266 | 0.211 | 1.000 | 1.000 |
| viol_traj | 0.494 | 0.505 | 1.000 | 1.000 |
| viol_final | 0.609 | 0.609 | 1.000 | 1.000 |
| unknown_final | 0.999 | 0.999 | 0.793 | 0.000 |
| util_final | 0.565 | 0.533 | 0.832 | 0.818 |

### Judge fitted on validation scenarios (test AUROC)

| predictors | auroc |
|---|---|
| final checks | 1.000 |
| final checks + identity | 1.000 |
| final checks + preference | 1.000 |
| final checks + identity + preference | 1.000 |

Paired scenario bootstrap (95% interval of the AUROC gain):

- final checks + identity over final checks: +0.000 [+0.000, +0.000]
- final checks + preference over final checks: +0.000 [+0.000, +0.001]
- final checks + identity + preference over final checks + identity: +0.000 [+0.000, +0.000]

### Diagnostic subsets (changing = membership change > 0.100, the validation median)

| subset | cases | mean regret | flagged by u_set | flagged by u_rank | flagged by u_attr | flagged by u_util |
|---|---|---|---|---|---|---|
| changing but suitable | 9 | 0.291 | 1.000 | 1.000 | 1.000 | 1.000 |
| consequentially unstable | 1005 | 0.954 | 1.000 | 0.999 | 1.000 | 1.000 |
| stable but unsuitable | 784 | 0.963 | 1.000 | 0.726 | 1.000 | 1.000 |
| stable and suitable | 2 | 0.412 | 1.000 | 1.000 | 1.000 | 1.000 |

## Pipeline: hard

Test detection rows: 1800 (14 failures).

### Single signals (test)

| signal | auroc | auroc_mean_over_runs | recall at matched threshold | false alarms among suitable |
|---|---|---|---|---|
| u_set | 0.347 | 0.365 | 0.000 | 0.000 |
| u_rank | 0.246 | 0.253 | 0.000 | 0.000 |
| u_attr | 0.427 | 0.468 | 0.000 | 0.000 |
| u_util | 0.434 | 0.474 | 0.000 | 0.000 |
| u_attr_rank | 0.451 | 0.483 | 0.000 | 0.000 |
| u_dist | 0.439 | 0.483 | 0.000 | 0.000 |
| u_generic | 0.376 | 0.412 | 0.000 | 0.000 |
| util_final | 0.963 | 0.968 | 0.000 | 0.000 |


### Diagnostic subsets (changing = membership change > 0.075, the validation median)

| subset | cases | mean regret | flagged by u_set | flagged by u_rank | flagged by u_attr | flagged by u_util |
|---|---|---|---|---|---|---|
| changing but suitable | 977 | 0.159 | 0.000 | 0.000 | 0.000 | 0.000 |
| consequentially unstable | 5 | 0.546 | 0.000 | 0.000 | 0.000 | 0.000 |
| stable but unsuitable | 9 | 0.572 | 0.000 | 0.000 | 0.000 | 0.000 |
| stable and suitable | 809 | 0.125 | 0.000 | 0.000 | 0.000 | 0.000 |
