# Proposal section 4: number of sampling steps (amazon_video_games, valid_risk)

5 models trained with 5 steps, sampled on subsets of their steps; 14,835 cases. Controls: log history length and final-score margin (5-fold CV, folds by user).

| Steps | Recall@10 | Final list same as 5 steps | Mean step change@10 | AUROC | Gain over controls (sd) |
|---|---|---|---|---|---|
| 5 | 0.0811 | 100.0% | 0.083 | 0.562 | +0.033 (0.016) |
| 4 | 0.0811 | 100.0% | 0.080 | 0.560 | +0.031 (0.016) |
| 3 | 0.0811 | 100.0% | 0.087 | 0.561 | +0.033 (0.016) |
| 2 | 0.0812 | 99.9% | 0.084 | 0.552 | +0.027 (0.015) |

Overlap of each checkpoint's top-10 with the final top-10, by normalized noise level (1 = noisiest visited step, 0 = final; mean over models):

| Normalized level | 2 steps | 3 steps | 4 steps | 5 steps |
|---|---|---|---|---|
| 1.00 | 0.916 | 0.916 | 0.916 | 0.916 |
| 0.75 |  |  | 0.915 | 0.915 |
| 0.50 |  | 0.909 |  | 0.909 |
| 0.25 |  |  | 0.930 | 0.930 |
| 0.00 | 1.000 | 1.000 | 1.000 | 1.000 |
