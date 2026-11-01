# Proposal section 4: tuned baselines and DiffRec on movielens

Settings chosen on valid_model (27,651 cases) by NDCG@10; each chosen method was then run once on test (110,457 cases).

## Test results

| Method | Setting | Recall@10 | NDCG@10 | Recall@50 |
|---|---|---|---|---|
| popularity |  | 0.0614 | 0.0307 | 0.1916 |
| co-occurrence |  | 0.0706 | 0.0354 | 0.2124 |
| conditional |  | 0.0730 | 0.0366 | 0.2205 |
| recent popularity | {"days": 90} | 0.0612 | 0.0303 | 0.1936 |
| itemknn | {"k": 20, "shrink": 0} | 0.0842 | 0.0413 | 0.2662 |
| ease | {"lam": 500} | 0.1019 | 0.0498 | 0.3136 |
| DiffRec | mean of 5 seeds | 0.0946 | 0.0467 | 0.2888 |

## Tuning on valid_model (NDCG@10)

| Method | Setting | NDCG@10 | Recall@10 |
|---|---|---|---|
| popularity | {} **(chosen)** | 0.0299 | 0.0604 |
| co-occurrence | {} **(chosen)** | 0.0361 | 0.0739 |
| conditional | {} **(chosen)** | 0.0370 | 0.0757 |
| recent popularity | {"days": 30} | 0.0301 | 0.0613 |
| recent popularity | {"days": 90} **(chosen)** | 0.0302 | 0.0613 |
| recent popularity | {"days": 180} | 0.0299 | 0.0605 |
| recent popularity | {"days": 365} | 0.0299 | 0.0604 |
| itemknn | {"k": 20, "shrink": 0} **(chosen)** | 0.0431 | 0.0899 |
| itemknn | {"k": 20, "shrink": 10} | 0.0430 | 0.0894 |
| itemknn | {"k": 20, "shrink": 100} | 0.0418 | 0.0859 |
| itemknn | {"k": 50, "shrink": 0} | 0.0420 | 0.0867 |
| itemknn | {"k": 50, "shrink": 10} | 0.0419 | 0.0861 |
| itemknn | {"k": 50, "shrink": 100} | 0.0403 | 0.0830 |
| itemknn | {"k": 100, "shrink": 0} | 0.0417 | 0.0861 |
| itemknn | {"k": 100, "shrink": 10} | 0.0414 | 0.0848 |
| itemknn | {"k": 100, "shrink": 100} | 0.0402 | 0.0827 |
| itemknn | {"k": 200, "shrink": 0} | 0.0421 | 0.0854 |
| itemknn | {"k": 200, "shrink": 10} | 0.0417 | 0.0848 |
| itemknn | {"k": 200, "shrink": 100} | 0.0407 | 0.0835 |
| itemknn | {"k": 500, "shrink": 0} | 0.0429 | 0.0878 |
| itemknn | {"k": 500, "shrink": 10} | 0.0427 | 0.0878 |
| itemknn | {"k": 500, "shrink": 100} | 0.0415 | 0.0849 |
| itemknn | {"k": null, "shrink": 0} | 0.0430 | 0.0883 |
| itemknn | {"k": null, "shrink": 10} | 0.0429 | 0.0877 |
| itemknn | {"k": null, "shrink": 100} | 0.0416 | 0.0852 |
| ease | {"lam": 50} | 0.0512 | 0.1059 |
| ease | {"lam": 200} | 0.0538 | 0.1133 |
| ease | {"lam": 500} **(chosen)** | 0.0548 | 0.1154 |
| ease | {"lam": 1000} | 0.0548 | 0.1145 |
| ease | {"lam": 3000} | 0.0530 | 0.1100 |
| ease | {"lam": 10000} | 0.0481 | 0.0986 |
