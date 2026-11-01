# Proposal section 4: tuned baselines and DiffRec on amazon_video_games

Settings chosen on valid_model (19,085 cases) by NDCG@10; each chosen method was then run once on test (26,350 cases).

## Test results

| Method | Setting | Recall@10 | NDCG@10 | Recall@50 |
|---|---|---|---|---|
| popularity |  | 0.0327 | 0.0145 | 0.0866 |
| co-occurrence |  | 0.0790 | 0.0448 | 0.1529 |
| conditional |  | 0.0793 | 0.0461 | 0.1552 |
| recent popularity | {"days": 365} | 0.0661 | 0.0359 | 0.1369 |
| itemknn | {"k": 500, "shrink": 100} | 0.0820 | 0.0472 | 0.1605 |
| ease | {"lam": 50} | 0.0862 | 0.0506 | 0.1554 |
| DiffRec | mean of 5 seeds | 0.0707 | 0.0402 | 0.1325 |

Warning: the chosen setting lies at the edge of the grid for recent popularity; the grid may need to be extended.

## Tuning on valid_model (NDCG@10)

| Method | Setting | NDCG@10 | Recall@10 |
|---|---|---|---|
| popularity | {} **(chosen)** | 0.0179 | 0.0412 |
| co-occurrence | {} **(chosen)** | 0.0527 | 0.0914 |
| conditional | {} **(chosen)** | 0.0586 | 0.0972 |
| recent popularity | {"days": 30} | 0.0352 | 0.0611 |
| recent popularity | {"days": 90} | 0.0374 | 0.0679 |
| recent popularity | {"days": 180} | 0.0372 | 0.0674 |
| recent popularity | {"days": 365} **(chosen)** | 0.0401 | 0.0738 |
| itemknn | {"k": 5, "shrink": 0} | 0.0388 | 0.0648 |
| itemknn | {"k": 5, "shrink": 10} | 0.0477 | 0.0795 |
| itemknn | {"k": 5, "shrink": 100} | 0.0538 | 0.0913 |
| itemknn | {"k": 5, "shrink": 300} | 0.0548 | 0.0938 |
| itemknn | {"k": 5, "shrink": 1000} | 0.0545 | 0.0937 |
| itemknn | {"k": 10, "shrink": 0} | 0.0400 | 0.0685 |
| itemknn | {"k": 10, "shrink": 10} | 0.0494 | 0.0849 |
| itemknn | {"k": 10, "shrink": 100} | 0.0556 | 0.0968 |
| itemknn | {"k": 10, "shrink": 300} | 0.0557 | 0.0958 |
| itemknn | {"k": 10, "shrink": 1000} | 0.0549 | 0.0941 |
| itemknn | {"k": 20, "shrink": 0} | 0.0406 | 0.0696 |
| itemknn | {"k": 20, "shrink": 10} | 0.0498 | 0.0850 |
| itemknn | {"k": 20, "shrink": 100} | 0.0556 | 0.0961 |
| itemknn | {"k": 20, "shrink": 300} | 0.0557 | 0.0961 |
| itemknn | {"k": 20, "shrink": 1000} | 0.0546 | 0.0938 |
| itemknn | {"k": 50, "shrink": 0} | 0.0412 | 0.0709 |
| itemknn | {"k": 50, "shrink": 10} | 0.0503 | 0.0864 |
| itemknn | {"k": 50, "shrink": 100} | 0.0560 | 0.0969 |
| itemknn | {"k": 50, "shrink": 300} | 0.0560 | 0.0967 |
| itemknn | {"k": 50, "shrink": 1000} | 0.0555 | 0.0959 |
| itemknn | {"k": 100, "shrink": 0} | 0.0420 | 0.0720 |
| itemknn | {"k": 100, "shrink": 10} | 0.0507 | 0.0870 |
| itemknn | {"k": 100, "shrink": 100} | 0.0566 | 0.0981 |
| itemknn | {"k": 100, "shrink": 300} | 0.0564 | 0.0978 |
| itemknn | {"k": 100, "shrink": 1000} | 0.0555 | 0.0961 |
| itemknn | {"k": 200, "shrink": 0} | 0.0432 | 0.0747 |
| itemknn | {"k": 200, "shrink": 10} | 0.0515 | 0.0891 |
| itemknn | {"k": 200, "shrink": 100} | 0.0569 | 0.0989 |
| itemknn | {"k": 200, "shrink": 300} | 0.0563 | 0.0973 |
| itemknn | {"k": 200, "shrink": 1000} | 0.0553 | 0.0953 |
| itemknn | {"k": 500, "shrink": 0} | 0.0438 | 0.0757 |
| itemknn | {"k": 500, "shrink": 10} | 0.0519 | 0.0898 |
| itemknn | {"k": 500, "shrink": 100} **(chosen)** | 0.0570 | 0.0986 |
| itemknn | {"k": 500, "shrink": 300} | 0.0565 | 0.0976 |
| itemknn | {"k": 500, "shrink": 1000} | 0.0553 | 0.0953 |
| itemknn | {"k": null, "shrink": 0} | 0.0440 | 0.0758 |
| itemknn | {"k": null, "shrink": 10} | 0.0520 | 0.0895 |
| itemknn | {"k": null, "shrink": 100} | 0.0568 | 0.0986 |
| itemknn | {"k": null, "shrink": 300} | 0.0565 | 0.0975 |
| itemknn | {"k": null, "shrink": 1000} | 0.0554 | 0.0958 |
| ease | {"lam": 5} | 0.0581 | 0.0947 |
| ease | {"lam": 10} | 0.0599 | 0.0978 |
| ease | {"lam": 20} | 0.0612 | 0.1007 |
| ease | {"lam": 50} **(chosen)** | 0.0618 | 0.1033 |
| ease | {"lam": 200} | 0.0607 | 0.1038 |
| ease | {"lam": 500} | 0.0591 | 0.1017 |
| ease | {"lam": 1000} | 0.0579 | 0.0997 |
| ease | {"lam": 3000} | 0.0562 | 0.0973 |
| ease | {"lam": 10000} | 0.0548 | 0.0950 |
| ease | {"lam": 30000} | 0.0541 | 0.0938 |
| ease | {"lam": 100000} | 0.0540 | 0.0935 |
