# Proposal section 4: tuned baselines and DiffRec on amazon_books

Settings chosen on valid_model (136,077 cases) by NDCG@10; each chosen method was then run once on test (253,631 cases).

## Test results

| Method | Setting | Recall@10 | NDCG@10 | Recall@50 |
|---|---|---|---|---|
| popularity |  | 0.0064 | 0.0031 | 0.0264 |
| co-occurrence |  | 0.1039 | 0.0638 | 0.1811 |
| conditional |  | 0.1162 | 0.0710 | 0.1962 |
| recent popularity | {"days": 30} | 0.0096 | 0.0050 | 0.0254 |
| itemknn | {"k": 10, "shrink": 100} | 0.1255 | 0.0769 | 0.1862 |
| ease | {"lam": 20} | 0.1306 | 0.0806 | 0.2001 |
| DiffRec | mean of 5 seeds | 0.0753 | 0.0439 | 0.1366 |

Warning: the chosen setting lies at the edge of the grid for recent popularity; the grid may need to be extended.

## Tuning on valid_model (NDCG@10)

| Method | Setting | NDCG@10 | Recall@10 |
|---|---|---|---|
| popularity | {} **(chosen)** | 0.0031 | 0.0068 |
| co-occurrence | {} **(chosen)** | 0.0669 | 0.1108 |
| conditional | {} **(chosen)** | 0.0746 | 0.1235 |
| recent popularity | {"days": 30} **(chosen)** | 0.0145 | 0.0275 |
| recent popularity | {"days": 90} | 0.0115 | 0.0211 |
| recent popularity | {"days": 180} | 0.0102 | 0.0212 |
| recent popularity | {"days": 365} | 0.0084 | 0.0170 |
| itemknn | {"k": 5, "shrink": 0} | 0.0783 | 0.1262 |
| itemknn | {"k": 5, "shrink": 10} | 0.0794 | 0.1278 |
| itemknn | {"k": 5, "shrink": 100} | 0.0797 | 0.1298 |
| itemknn | {"k": 5, "shrink": 300} | 0.0776 | 0.1276 |
| itemknn | {"k": 5, "shrink": 1000} | 0.0743 | 0.1229 |
| itemknn | {"k": 10, "shrink": 0} | 0.0792 | 0.1298 |
| itemknn | {"k": 10, "shrink": 10} | 0.0804 | 0.1318 |
| itemknn | {"k": 10, "shrink": 100} **(chosen)** | 0.0805 | 0.1329 |
| itemknn | {"k": 10, "shrink": 300} | 0.0779 | 0.1290 |
| itemknn | {"k": 10, "shrink": 1000} | 0.0740 | 0.1231 |
| itemknn | {"k": 20, "shrink": 0} | 0.0786 | 0.1280 |
| itemknn | {"k": 20, "shrink": 10} | 0.0798 | 0.1300 |
| itemknn | {"k": 20, "shrink": 100} | 0.0800 | 0.1315 |
| itemknn | {"k": 20, "shrink": 300} | 0.0773 | 0.1274 |
| itemknn | {"k": 20, "shrink": 1000} | 0.0733 | 0.1210 |
| itemknn | {"k": 50, "shrink": 0} | 0.0786 | 0.1279 |
| itemknn | {"k": 50, "shrink": 10} | 0.0797 | 0.1299 |
| itemknn | {"k": 50, "shrink": 100} | 0.0799 | 0.1312 |
| itemknn | {"k": 50, "shrink": 300} | 0.0771 | 0.1269 |
| itemknn | {"k": 50, "shrink": 1000} | 0.0730 | 0.1205 |
| itemknn | {"k": 100, "shrink": 0} | 0.0785 | 0.1278 |
| itemknn | {"k": 100, "shrink": 10} | 0.0796 | 0.1296 |
| itemknn | {"k": 100, "shrink": 100} | 0.0798 | 0.1312 |
| itemknn | {"k": 100, "shrink": 300} | 0.0769 | 0.1264 |
| itemknn | {"k": 100, "shrink": 1000} | 0.0727 | 0.1197 |
| itemknn | {"k": 200, "shrink": 0} | 0.0785 | 0.1281 |
| itemknn | {"k": 200, "shrink": 10} | 0.0797 | 0.1299 |
| itemknn | {"k": 200, "shrink": 100} | 0.0799 | 0.1311 |
| itemknn | {"k": 200, "shrink": 300} | 0.0768 | 0.1261 |
| itemknn | {"k": 200, "shrink": 1000} | 0.0728 | 0.1199 |
| itemknn | {"k": 500, "shrink": 0} | 0.0788 | 0.1285 |
| itemknn | {"k": 500, "shrink": 10} | 0.0799 | 0.1303 |
| itemknn | {"k": 500, "shrink": 100} | 0.0800 | 0.1311 |
| itemknn | {"k": 500, "shrink": 300} | 0.0769 | 0.1263 |
| itemknn | {"k": 500, "shrink": 1000} | 0.0729 | 0.1202 |
| itemknn | {"k": null, "shrink": 0} | 0.0789 | 0.1288 |
| itemknn | {"k": null, "shrink": 10} | 0.0800 | 0.1307 |
| itemknn | {"k": null, "shrink": 100} | 0.0800 | 0.1314 |
| itemknn | {"k": null, "shrink": 300} | 0.0769 | 0.1264 |
| itemknn | {"k": null, "shrink": 1000} | 0.0729 | 0.1202 |
| ease | {"lam": 5} | 0.0845 | 0.1363 |
| ease | {"lam": 10} | 0.0851 | 0.1377 |
| ease | {"lam": 20} **(chosen)** | 0.0855 | 0.1393 |
| ease | {"lam": 50} | 0.0855 | 0.1407 |
| ease | {"lam": 200} | 0.0828 | 0.1382 |
| ease | {"lam": 500} | 0.0792 | 0.1325 |
| ease | {"lam": 1000} | 0.0765 | 0.1278 |
| ease | {"lam": 3000} | 0.0729 | 0.1213 |
| ease | {"lam": 10000} | 0.0700 | 0.1162 |
| ease | {"lam": 30000} | 0.0687 | 0.1139 |
| ease | {"lam": 100000} | 0.0681 | 0.1128 |
