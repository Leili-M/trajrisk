# Proposal section 4: number of sampling steps (amazon_books, valid_risk)

5 models trained with 5 steps, sampled on subsets of their steps; 104,919 cases. Controls: log history length, final-score margin and mean popularity of the history items, the three controls of the main analysis (5-fold CV, folds by user).

| Steps | Recall@10 | Final list same as 5 steps | Mean step change@10 | AUROC | Gain over controls (sd) |
|---|---|---|---|---|---|
| 5 | 0.0804 | 100.0% | 0.046 | 0.595 | +0.010 (0.004) |
| 4 | 0.0804 | 100.0% | 0.044 | 0.592 | +0.009 (0.003) |
| 3 | 0.0804 | 100.0% | 0.048 | 0.592 | +0.010 (0.004) |
| 2 | 0.0804 | 99.9% | 0.046 | 0.577 | +0.006 (0.003) |

Overlap of each checkpoint's top-10 with the final top-10, by normalized noise level (1 = noisiest visited step, 0 = final; mean over models):

| Normalized level | 2 steps | 3 steps | 4 steps | 5 steps |
|---|---|---|---|---|
| 1.00 | 0.954 | 0.954 | 0.954 | 0.954 |
| 0.75 |  |  | 0.950 | 0.950 |
| 0.50 |  | 0.949 |  | 0.949 |
| 0.25 |  |  | 0.963 | 0.963 |
| 0.00 | 1.000 | 1.000 | 1.000 | 1.000 |
