# Proposal section 4: tuned baselines and DiffRec on base

Settings chosen on valid_model (24,317 cases) by NDCG@10; each chosen method was then run once on test (60,458 cases).

## Test results

| Method | Setting | Recall@10 | NDCG@10 | Recall@50 |
|---|---|---|---|---|
| popularity |  | 0.0192 | 0.0096 | 0.0536 |
| co-occurrence |  | 0.0165 | 0.0086 | 0.0468 |
| conditional |  | 0.0148 | 0.0077 | 0.0428 |
| recent popularity | {"days": 365} | 0.0190 | 0.0094 | 0.0577 |
| itemknn | {"k": 200, "shrink": 1000} | 0.0159 | 0.0082 | 0.0455 |
| ease | {"lam": 100000} | 0.0162 | 0.0084 | 0.0464 |
| DiffRec | mean of 5 seeds | 0.0157 | 0.0081 | 0.0401 |

Warning: the chosen setting lies at the edge of the grid for recent popularity, itemknn, ease; the grid may need to be extended.

## Tuning on valid_model (NDCG@10)

| Method | Setting | NDCG@10 | Recall@10 |
|---|---|---|---|
| popularity | {} **(chosen)** | 0.0103 | 0.0192 |
| co-occurrence | {} **(chosen)** | 0.0093 | 0.0175 |
| conditional | {} **(chosen)** | 0.0087 | 0.0167 |
| recent popularity | {"days": 30} | 0.0023 | 0.0051 |
| recent popularity | {"days": 90} | 0.0069 | 0.0159 |
| recent popularity | {"days": 180} | 0.0097 | 0.0190 |
| recent popularity | {"days": 365} **(chosen)** | 0.0106 | 0.0202 |
| itemknn | {"k": 5, "shrink": 0} | 0.0010 | 0.0019 |
| itemknn | {"k": 5, "shrink": 10} | 0.0029 | 0.0049 |
| itemknn | {"k": 5, "shrink": 100} | 0.0066 | 0.0124 |
| itemknn | {"k": 5, "shrink": 300} | 0.0078 | 0.0145 |
| itemknn | {"k": 5, "shrink": 1000} | 0.0084 | 0.0155 |
| itemknn | {"k": 10, "shrink": 0} | 0.0015 | 0.0028 |
| itemknn | {"k": 10, "shrink": 10} | 0.0037 | 0.0063 |
| itemknn | {"k": 10, "shrink": 100} | 0.0071 | 0.0134 |
| itemknn | {"k": 10, "shrink": 300} | 0.0080 | 0.0150 |
| itemknn | {"k": 10, "shrink": 1000} | 0.0087 | 0.0162 |
| itemknn | {"k": 20, "shrink": 0} | 0.0018 | 0.0032 |
| itemknn | {"k": 20, "shrink": 10} | 0.0038 | 0.0069 |
| itemknn | {"k": 20, "shrink": 100} | 0.0071 | 0.0134 |
| itemknn | {"k": 20, "shrink": 300} | 0.0083 | 0.0158 |
| itemknn | {"k": 20, "shrink": 1000} | 0.0087 | 0.0162 |
| itemknn | {"k": 50, "shrink": 0} | 0.0020 | 0.0038 |
| itemknn | {"k": 50, "shrink": 10} | 0.0040 | 0.0075 |
| itemknn | {"k": 50, "shrink": 100} | 0.0075 | 0.0143 |
| itemknn | {"k": 50, "shrink": 300} | 0.0082 | 0.0153 |
| itemknn | {"k": 50, "shrink": 1000} | 0.0088 | 0.0164 |
| itemknn | {"k": 100, "shrink": 0} | 0.0018 | 0.0034 |
| itemknn | {"k": 100, "shrink": 10} | 0.0037 | 0.0074 |
| itemknn | {"k": 100, "shrink": 100} | 0.0072 | 0.0137 |
| itemknn | {"k": 100, "shrink": 300} | 0.0082 | 0.0155 |
| itemknn | {"k": 100, "shrink": 1000} | 0.0088 | 0.0165 |
| itemknn | {"k": 200, "shrink": 0} | 0.0018 | 0.0036 |
| itemknn | {"k": 200, "shrink": 10} | 0.0037 | 0.0074 |
| itemknn | {"k": 200, "shrink": 100} | 0.0073 | 0.0138 |
| itemknn | {"k": 200, "shrink": 300} | 0.0083 | 0.0156 |
| itemknn | {"k": 200, "shrink": 1000} **(chosen)** | 0.0089 | 0.0168 |
| itemknn | {"k": 500, "shrink": 0} | 0.0016 | 0.0031 |
| itemknn | {"k": 500, "shrink": 10} | 0.0037 | 0.0074 |
| itemknn | {"k": 500, "shrink": 100} | 0.0073 | 0.0139 |
| itemknn | {"k": 500, "shrink": 300} | 0.0084 | 0.0159 |
| itemknn | {"k": 500, "shrink": 1000} | 0.0088 | 0.0164 |
| itemknn | {"k": null, "shrink": 0} | 0.0027 | 0.0057 |
| itemknn | {"k": null, "shrink": 10} | 0.0044 | 0.0089 |
| itemknn | {"k": null, "shrink": 100} | 0.0073 | 0.0140 |
| itemknn | {"k": null, "shrink": 300} | 0.0085 | 0.0161 |
| itemknn | {"k": null, "shrink": 1000} | 0.0089 | 0.0165 |
| ease | {"lam": 5} | 0.0047 | 0.0086 |
| ease | {"lam": 10} | 0.0056 | 0.0107 |
| ease | {"lam": 20} | 0.0065 | 0.0126 |
| ease | {"lam": 50} | 0.0073 | 0.0139 |
| ease | {"lam": 200} | 0.0083 | 0.0160 |
| ease | {"lam": 500} | 0.0085 | 0.0162 |
| ease | {"lam": 1000} | 0.0087 | 0.0167 |
| ease | {"lam": 3000} | 0.0088 | 0.0170 |
| ease | {"lam": 10000} | 0.0091 | 0.0172 |
| ease | {"lam": 30000} | 0.0091 | 0.0172 |
| ease | {"lam": 100000} **(chosen)** | 0.0092 | 0.0171 |
