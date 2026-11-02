# Section 6: preference-aware comparison (HUMMUS benchmark, DiffRec)

Runs: hm_seed42, hm_seed1, hm_seed2, hm_seed3, hm_seed4. Reranker (formula 9) chosen on validation scenarios: beta = 8, gamma = 20. Failure = regret >= 0.5. Detection rows: scenarios with feasible recipes; thresholds and judge from validation.

## Reranker grid (validation, soft pipeline)

| beta | gamma | valid mean regret |
|---|---|---|
| 0.000 | 0.000 | 0.995 |
| 0.000 | 0.250 | 0.419 |
| 0.000 | 0.500 | 0.324 |
| 0.000 | 1.000 | 0.284 |
| 0.000 | 2.000 | 0.284 |
| 0.000 | 5.000 | 0.284 |
| 0.000 | 20.000 | 0.284 |
| 0.250 | 0.000 | 0.980 |
| 0.250 | 0.250 | 0.307 |
| 0.250 | 0.500 | 0.193 |
| 0.250 | 1.000 | 0.144 |
| 0.250 | 2.000 | 0.142 |
| 0.250 | 5.000 | 0.142 |
| 0.250 | 20.000 | 0.142 |
| 0.500 | 0.000 | 0.981 |
| 0.500 | 0.250 | 0.319 |
| 0.500 | 0.500 | 0.179 |
| 0.500 | 1.000 | 0.114 |
| 0.500 | 2.000 | 0.112 |
| 0.500 | 5.000 | 0.112 |
| 0.500 | 20.000 | 0.112 |
| 1.000 | 0.000 | 0.983 |
| 1.000 | 0.250 | 0.387 |
| 1.000 | 0.500 | 0.202 |
| 1.000 | 1.000 | 0.089 |
| 1.000 | 2.000 | 0.080 |
| 1.000 | 5.000 | 0.080 |
| 1.000 | 20.000 | 0.080 |
| 2.000 | 0.000 | 0.984 |
| 2.000 | 0.250 | 0.534 |
| 2.000 | 0.500 | 0.286 |
| 2.000 | 1.000 | 0.090 |
| 2.000 | 2.000 | 0.049 |
| 2.000 | 5.000 | 0.048 |
| 2.000 | 20.000 | 0.048 |
| 4.000 | 0.000 | 0.985 |
| 4.000 | 0.250 | 0.691 |
| 4.000 | 0.500 | 0.461 |
| 4.000 | 1.000 | 0.176 |
| 4.000 | 2.000 | 0.043 |
| 4.000 | 5.000 | 0.024 |
| 4.000 | 20.000 | 0.024 |
| 8.000 | 0.000 | 0.985 |
| 8.000 | 0.250 | 0.787 |
| 8.000 | 0.500 | 0.633 |
| 8.000 | 1.000 | 0.380 |
| 8.000 | 2.000 | 0.111 |
| 8.000 | 5.000 | 0.014 |
| 8.000 | 20.000 | 0.010 |

## Outcomes on test

| pipeline | mean regret (many) | mean regret (few) | failure rate | lists with a violation | abstained when nothing feasible | regret, held-out combinations | regret, other | diversity |
|---|---|---|---|---|---|---|---|---|
| hard | 0.013 | 0.007 | 0.000 | 0.000 | 1.000 | 0.010 | 0.010 | 0.191 |
| plain | 0.990 | 1.000 | 1.000 | 1.000 | 0.000 | 0.995 | 0.995 | 0.298 |
| soft | 0.013 | 0.007 | 0.000 | 0.000 | 0.000 | 0.010 | 0.010 | 0.191 |

## Pipeline: soft

Test detection rows: 1800 (0 failures).

### Agreement with list quality (test): Spearman correlation with regret

| signal | spearman with regret |
|---|---|
| u_set | 0.298 |
| u_rank | 0.296 |
| u_attr | 0.417 |
| u_util | 0.416 |
| u_attr_rank | 0.403 |
| u_dist | 0.446 |
| u_generic | 0.303 |
| util_final | -0.129 |

### Flags at a fixed rate: each signal flags lists above its validation 75th percentile (test). False alarm = flagged list that is suitable (regret < 0.5).

| signal | flag rate | suitable lists flagged | flags that are false alarms | regret flagged | regret not flagged | failures flagged |
|---|---|---|---|---|---|---|
| u_set | 0.155 | 0.155 | 1.000 | 0.014 | 0.009 | nan |
| u_rank | 0.319 | 0.319 | 1.000 | 0.012 | 0.009 | nan |
| u_attr | 0.306 | 0.306 | 1.000 | 0.014 | 0.008 | nan |
| u_util | 0.304 | 0.304 | 1.000 | 0.014 | 0.008 | nan |
| u_attr_rank | 0.302 | 0.302 | 1.000 | 0.013 | 0.009 | nan |
| u_dist | 0.294 | 0.294 | 1.000 | 0.014 | 0.008 | nan |
| u_generic | 0.306 | 0.306 | 1.000 | 0.013 | 0.009 | nan |
| util_final | 0.238 | 0.238 | 1.000 | 0.008 | 0.011 | nan |

### Single signals (test)

| signal | auroc | auroc_mean_over_runs | recall at matched threshold | false alarms among suitable |
|---|---|---|---|---|
| u_set | nan | nan | nan | 0.000 |
| u_rank | nan | nan | nan | 0.000 |
| u_attr | nan | nan | nan | 0.000 |
| u_util | nan | nan | nan | 0.000 |
| u_attr_rank | nan | nan | nan | 0.000 |
| u_dist | nan | nan | nan | 0.000 |
| u_generic | nan | nan | nan | 0.000 |
| util_final | nan | nan | nan | 0.000 |


### Diagnostic subsets (changing = membership change > 0.000, the validation median; flags at the fixed-rate thresholds above)

| subset | cases | mean regret | flagged by u_set | flagged by u_rank | flagged by u_attr | flagged by u_util |
|---|---|---|---|---|---|---|
| changing but suitable | 862 | 0.012 | 0.324 | 0.570 | 0.638 | 0.635 |
| consequentially unstable | 0 | nan | nan | nan | nan | nan |
| stable but unsuitable | 0 | nan | nan | nan | nan | nan |
| stable and suitable | 938 | 0.008 | 0.000 | 0.088 | 0.000 | 0.000 |

## Pipeline: hard

Test detection rows: 1800 (0 failures).

### Agreement with list quality (test): Spearman correlation with regret

| signal | spearman with regret |
|---|---|
| u_set | 0.298 |
| u_rank | 0.296 |
| u_attr | 0.417 |
| u_util | 0.416 |
| u_attr_rank | 0.403 |
| u_dist | 0.446 |
| u_generic | 0.303 |
| util_final | -0.129 |

### Flags at a fixed rate: each signal flags lists above its validation 75th percentile (test). False alarm = flagged list that is suitable (regret < 0.5).

| signal | flag rate | suitable lists flagged | flags that are false alarms | regret flagged | regret not flagged | failures flagged |
|---|---|---|---|---|---|---|
| u_set | 0.155 | 0.155 | 1.000 | 0.014 | 0.009 | nan |
| u_rank | 0.319 | 0.319 | 1.000 | 0.012 | 0.009 | nan |
| u_attr | 0.306 | 0.306 | 1.000 | 0.014 | 0.008 | nan |
| u_util | 0.304 | 0.304 | 1.000 | 0.014 | 0.008 | nan |
| u_attr_rank | 0.302 | 0.302 | 1.000 | 0.013 | 0.009 | nan |
| u_dist | 0.294 | 0.294 | 1.000 | 0.014 | 0.008 | nan |
| u_generic | 0.306 | 0.306 | 1.000 | 0.013 | 0.009 | nan |
| util_final | 0.238 | 0.238 | 1.000 | 0.008 | 0.011 | nan |

### Single signals (test)

| signal | auroc | auroc_mean_over_runs | recall at matched threshold | false alarms among suitable |
|---|---|---|---|---|
| u_set | nan | nan | nan | 0.000 |
| u_rank | nan | nan | nan | 0.000 |
| u_attr | nan | nan | nan | 0.000 |
| u_util | nan | nan | nan | 0.000 |
| u_attr_rank | nan | nan | nan | 0.000 |
| u_dist | nan | nan | nan | 0.000 |
| u_generic | nan | nan | nan | 0.000 |
| util_final | nan | nan | nan | 0.000 |


### Diagnostic subsets (changing = membership change > 0.000, the validation median; flags at the fixed-rate thresholds above)

| subset | cases | mean regret | flagged by u_set | flagged by u_rank | flagged by u_attr | flagged by u_util |
|---|---|---|---|---|---|---|
| changing but suitable | 862 | 0.012 | 0.324 | 0.570 | 0.638 | 0.635 |
| consequentially unstable | 0 | nan | nan | nan | nan | nan |
| stable but unsuitable | 0 | nan | nan | nan | nan | nan |
| stable and suitable | 938 | 0.008 | 0.000 | 0.088 | 0.000 | 0.000 |
