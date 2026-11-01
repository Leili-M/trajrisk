# Proposal section 4: number of sampling steps (movielens, valid_risk)

5 models trained with 5 steps, sampled on subsets of their steps; 26,399 cases. Controls: log history length, final-score margin and mean popularity of the history items, the three controls of the main analysis (5-fold CV, folds by user).

| Steps | Recall@10 | Final list same as 5 steps | Mean step change@10 | AUROC | Gain over controls (sd) |
|---|---|---|---|---|---|
| 5 | 0.1057 | 100.0% | 0.058 | 0.456 | +0.000 (0.000) |
| 4 | 0.1057 | 100.0% | 0.055 | 0.458 | +0.000 (0.000) |
| 3 | 0.1057 | 100.0% | 0.061 | 0.458 | +0.000 (0.000) |
| 2 | 0.1056 | 99.9% | 0.057 | 0.468 | -0.000 (0.000) |

Overlap of each checkpoint's top-10 with the final top-10, by normalized noise level (1 = noisiest visited step, 0 = final; mean over models):

| Normalized level | 2 steps | 3 steps | 4 steps | 5 steps |
|---|---|---|---|---|
| 1.00 | 0.943 | 0.943 | 0.943 | 0.943 |
| 0.75 |  |  | 0.932 | 0.932 |
| 0.50 |  | 0.935 |  | 0.935 |
| 0.25 |  |  | 0.960 | 0.960 |
| 0.00 | 1.000 | 1.000 | 1.000 | 1.000 |
