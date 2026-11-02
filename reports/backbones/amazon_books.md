# Proposal section 4: backbones on amazon_books

Same cases, masking and metrics as reports/baselines (mean and sd over runs).

| Backbone | Split | Runs | Recall@10 | NDCG@10 | Recall@50 |
|---|---|---|---|---|---|
| DiffRec | test | 5.0 | 0.0753 (0.0003) | 0.0439 (0.0002) | 0.1366 |
| DiffRec | valid_model | 5.0 | 0.0780 (0.0004) | 0.0449 (0.0002) | 0.1490 |
| GiffCF | test | 3.0 | 0.1257 (0.0002) | 0.0775 (0.0001) | 0.2001 |
| GiffCF | valid_model | 3.0 | 0.1333 (0.0001) | 0.0812 (0.0001) | 0.2211 |

## Per run

| Run | Backbone | Split | Recall@10 | NDCG@10 | Recall@50 |
|---|---|---|---|---|---|
| bk_seed42 | DiffRec | valid_model | 0.0782 | 0.0450 | 0.1495 |
| bk_seed42 | DiffRec | test | 0.0755 | 0.0440 | 0.1363 |
| bk_seed1 | DiffRec | valid_model | 0.0785 | 0.0452 | 0.1490 |
| bk_seed1 | DiffRec | test | 0.0750 | 0.0438 | 0.1364 |
| bk_seed2 | DiffRec | valid_model | 0.0777 | 0.0448 | 0.1484 |
| bk_seed2 | DiffRec | test | 0.0753 | 0.0437 | 0.1370 |
| bk_seed3 | DiffRec | valid_model | 0.0777 | 0.0448 | 0.1491 |
| bk_seed3 | DiffRec | test | 0.0751 | 0.0439 | 0.1366 |
| bk_seed4 | DiffRec | valid_model | 0.0781 | 0.0449 | 0.1489 |
| bk_seed4 | DiffRec | test | 0.0758 | 0.0441 | 0.1369 |
| bk_giff_s1 | GiffCF | valid_model | 0.1333 | 0.0812 | 0.2188 |
| bk_giff_s1 | GiffCF | test | 0.1255 | 0.0775 | 0.1979 |
| bk_giff_s2 | GiffCF | valid_model | 0.1334 | 0.0811 | 0.2242 |
| bk_giff_s2 | GiffCF | test | 0.1259 | 0.0774 | 0.2027 |
| bk_giff_s3 | GiffCF | valid_model | 0.1332 | 0.0812 | 0.2202 |
| bk_giff_s3 | GiffCF | test | 0.1258 | 0.0776 | 0.1997 |
