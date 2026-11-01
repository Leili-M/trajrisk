# Proposal section 4: number of sampling steps (base, valid_risk)

5 models trained with 5 steps, sampled on subsets of their steps; 23,891 cases. Controls: log history length, final-score margin and mean popularity of the history items, the three controls of the main analysis (5-fold CV, folds by user).

| Steps | Recall@10 | Final list same as 5 steps | Mean step change@10 | AUROC | Gain over controls (sd) |
|---|---|---|---|---|---|
| 5 | 0.0163 | 100.0% | 0.068 | 0.467 | +0.001 (0.002) |
| 4 | 0.0163 | 100.0% | 0.066 | 0.469 | +0.001 (0.002) |
| 3 | 0.0163 | 100.0% | 0.073 | 0.470 | +0.001 (0.002) |
| 2 | 0.0163 | 99.9% | 0.074 | 0.479 | +0.001 (0.002) |

Overlap of each checkpoint's top-10 with the final top-10, by normalized noise level (1 = noisiest visited step, 0 = final; mean over models):

| Normalized level | 2 steps | 3 steps | 4 steps | 5 steps |
|---|---|---|---|---|
| 1.00 | 0.926 | 0.926 | 0.926 | 0.926 |
| 0.75 |  |  | 0.926 | 0.925 |
| 0.50 |  | 0.929 |  | 0.929 |
| 0.25 |  |  | 0.950 | 0.950 |
| 0.00 | 1.000 | 1.000 | 1.000 | 1.000 |
