# Proposal section 4: backbones on amazon_video_games

Same cases, masking and metrics as reports/baselines (mean and sd over runs).

| Backbone | Split | Runs | Recall@10 | NDCG@10 | Recall@50 |
|---|---|---|---|---|---|
| DiffRec | test | 5.0 | 0.0707 (0.0011) | 0.0402 (0.0005) | 0.1325 |
| DiffRec | valid_model | 5.0 | 0.0867 (0.0014) | 0.0514 (0.0004) | 0.1585 |
| GiffCF | test | 3.0 | 0.0661 (0.0035) | 0.0384 (0.0017) | 0.1372 |
| GiffCF | valid_model | 3.0 | 0.0816 (0.0053) | 0.0470 (0.0030) | 0.1673 |

## Per run

| Run | Backbone | Split | Recall@10 | NDCG@10 | Recall@50 |
|---|---|---|---|---|---|
| vg_base | DiffRec | valid_model | 0.0850 | 0.0509 | 0.1559 |
| vg_base | DiffRec | test | 0.0704 | 0.0399 | 0.1311 |
| vg_seed1 | DiffRec | valid_model | 0.0876 | 0.0515 | 0.1592 |
| vg_seed1 | DiffRec | test | 0.0717 | 0.0409 | 0.1338 |
| vg_seed2 | DiffRec | valid_model | 0.0886 | 0.0518 | 0.1590 |
| vg_seed2 | DiffRec | test | 0.0715 | 0.0403 | 0.1319 |
| vg_seed3 | DiffRec | valid_model | 0.0859 | 0.0509 | 0.1577 |
| vg_seed3 | DiffRec | test | 0.0689 | 0.0395 | 0.1306 |
| vg_seed4 | DiffRec | valid_model | 0.0864 | 0.0517 | 0.1609 |
| vg_seed4 | DiffRec | test | 0.0711 | 0.0405 | 0.1354 |
| vg_giff_s1 | GiffCF | valid_model | 0.0786 | 0.0455 | 0.1606 |
| vg_giff_s1 | GiffCF | test | 0.0640 | 0.0374 | 0.1322 |
| vg_giff_s2 | GiffCF | valid_model | 0.0784 | 0.0450 | 0.1591 |
| vg_giff_s2 | GiffCF | test | 0.0642 | 0.0375 | 0.1305 |
| vg_giff_s3 | GiffCF | valid_model | 0.0878 | 0.0505 | 0.1821 |
| vg_giff_s3 | GiffCF | test | 0.0702 | 0.0403 | 0.1491 |
