# Proposal section 4: tuned baselines and DiffRec on movielens

Settings chosen on valid_model (27,651 cases) by NDCG@10; each chosen method was then run once on test (110,457 cases).

## Test results

| Method | Setting | Recall@10 | NDCG@10 | Recall@50 |
|---|---|---|---|---|
| popularity |  | 0.0614 | 0.0307 | 0.1916 |
| co-occurrence |  | 0.0706 | 0.0354 | 0.2124 |
| conditional |  | 0.0730 | 0.0366 | 0.2205 |
| recent popularity | {"days": 90} | 0.0612 | 0.0303 | 0.1936 |
| itemknn | {"k": 10, "shrink": 0} | 0.0865 | 0.0424 | 0.2592 |
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
| itemknn | {"k": 5, "shrink": 0} | 0.0442 | 0.0919 |
| itemknn | {"k": 5, "shrink": 10} | 0.0443 | 0.0914 |
| itemknn | {"k": 5, "shrink": 100} | 0.0431 | 0.0891 |
| itemknn | {"k": 5, "shrink": 300} | 0.0415 | 0.0854 |
| itemknn | {"k": 5, "shrink": 1000} | 0.0388 | 0.0796 |
| itemknn | {"k": 10, "shrink": 0} **(chosen)** | 0.0450 | 0.0930 |
| itemknn | {"k": 10, "shrink": 10} | 0.0448 | 0.0928 |
| itemknn | {"k": 10, "shrink": 100} | 0.0428 | 0.0884 |
| itemknn | {"k": 10, "shrink": 300} | 0.0407 | 0.0833 |
| itemknn | {"k": 10, "shrink": 1000} | 0.0382 | 0.0771 |
| itemknn | {"k": 20, "shrink": 0} | 0.0431 | 0.0899 |
| itemknn | {"k": 20, "shrink": 10} | 0.0430 | 0.0894 |
| itemknn | {"k": 20, "shrink": 100} | 0.0418 | 0.0859 |
| itemknn | {"k": 20, "shrink": 300} | 0.0400 | 0.0821 |
| itemknn | {"k": 20, "shrink": 1000} | 0.0374 | 0.0778 |
| itemknn | {"k": 50, "shrink": 0} | 0.0420 | 0.0867 |
| itemknn | {"k": 50, "shrink": 10} | 0.0419 | 0.0861 |
| itemknn | {"k": 50, "shrink": 100} | 0.0403 | 0.0830 |
| itemknn | {"k": 50, "shrink": 300} | 0.0391 | 0.0799 |
| itemknn | {"k": 50, "shrink": 1000} | 0.0377 | 0.0770 |
| itemknn | {"k": 100, "shrink": 0} | 0.0417 | 0.0861 |
| itemknn | {"k": 100, "shrink": 10} | 0.0414 | 0.0848 |
| itemknn | {"k": 100, "shrink": 100} | 0.0402 | 0.0827 |
| itemknn | {"k": 100, "shrink": 300} | 0.0394 | 0.0804 |
| itemknn | {"k": 100, "shrink": 1000} | 0.0382 | 0.0777 |
| itemknn | {"k": 200, "shrink": 0} | 0.0421 | 0.0854 |
| itemknn | {"k": 200, "shrink": 10} | 0.0417 | 0.0848 |
| itemknn | {"k": 200, "shrink": 100} | 0.0407 | 0.0835 |
| itemknn | {"k": 200, "shrink": 300} | 0.0401 | 0.0818 |
| itemknn | {"k": 200, "shrink": 1000} | 0.0384 | 0.0782 |
| itemknn | {"k": 500, "shrink": 0} | 0.0429 | 0.0878 |
| itemknn | {"k": 500, "shrink": 10} | 0.0427 | 0.0878 |
| itemknn | {"k": 500, "shrink": 100} | 0.0415 | 0.0849 |
| itemknn | {"k": 500, "shrink": 300} | 0.0404 | 0.0825 |
| itemknn | {"k": 500, "shrink": 1000} | 0.0383 | 0.0779 |
| itemknn | {"k": null, "shrink": 0} | 0.0430 | 0.0883 |
| itemknn | {"k": null, "shrink": 10} | 0.0429 | 0.0877 |
| itemknn | {"k": null, "shrink": 100} | 0.0416 | 0.0852 |
| itemknn | {"k": null, "shrink": 300} | 0.0404 | 0.0823 |
| itemknn | {"k": null, "shrink": 1000} | 0.0383 | 0.0779 |
| ease | {"lam": 5} | 0.0442 | 0.0896 |
| ease | {"lam": 10} | 0.0463 | 0.0944 |
| ease | {"lam": 20} | 0.0485 | 0.0995 |
| ease | {"lam": 50} | 0.0512 | 0.1059 |
| ease | {"lam": 200} | 0.0538 | 0.1133 |
| ease | {"lam": 500} **(chosen)** | 0.0548 | 0.1154 |
| ease | {"lam": 1000} | 0.0548 | 0.1145 |
| ease | {"lam": 3000} | 0.0530 | 0.1100 |
| ease | {"lam": 10000} | 0.0481 | 0.0986 |
| ease | {"lam": 30000} | 0.0429 | 0.0863 |
| ease | {"lam": 100000} | 0.0387 | 0.0774 |
