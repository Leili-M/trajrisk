# Section 9: routing to EASE, amazon_books (GiffCF)

Threshold chosen on risk-validation (cross-fitted judge), applied once to test. EASE lambda 20 (section 4). Mean over runs [95% t-interval]. Quality is over ALL test cases.

| Policy | NDCG@10 | Recall@10 | Routed | Gain on routed (NDCG) | Random, same rate | Coverage | Mean log-pop | Tail share | ms/case |
|---|---|---|---|---|---|---|---|---|---|
| always diffusion | 0.0775 [0.0773, 0.0777] | 0.1257 | 0.00 | – | – | 0.983 | 4.41 | 0.330 | 0.398 |
| always EASE | 0.0806 [0.0806, 0.0806] | 0.1306 | 1.00 | – | – | 0.939 | 5.15 | 0.127 | 0.150 |
| judge: controls + lists | 0.0806 [0.0806, 0.0806] | 0.1306 | 1.00 | +0.0031 | 0.0806 | 0.939 | 5.15 | 0.127 | 0.548 |
| judge: controls | 0.0806 [0.0806, 0.0806] | 0.1306 | 1.00 | +0.0031 | 0.0806 | 0.939 | 5.15 | 0.127 | 0.548 |
| entropy | 0.0806 [0.0806, 0.0806] | 0.1306 | 1.00 | +0.0031 | 0.0806 | 0.939 | 5.15 | 0.127 | 0.548 |

## Routing rate by history length (test, mean over runs)

| Policy | 1 | 2-3 | 4-9 | 10-29 | 30+ |
|---|---|---|---|---|---|
| judge: controls + lists | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| judge: controls | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| entropy | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

Fallback cost: build 15.94 s, 0.1503 ms/case (p95 batch 0.1516), peak GPU memory 52.5 GB. Diffusion cost per case is the logged trajectory run (includes logging, an upper bound). The routed cost always includes the full diffusion run, because the risk needs the completed trajectory.

Quality-rate curves: `<dataset>_curve.csv` (policies, random, oracle).
