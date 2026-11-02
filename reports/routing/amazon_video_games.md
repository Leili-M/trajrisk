# Section 9: routing to EASE, amazon_video_games (DiffRec)

Threshold chosen on risk-validation (cross-fitted judge), applied once to test. EASE lambda 50 (section 4). Mean over runs [95% t-interval]. Quality is over ALL test cases.

| Policy | NDCG@10 | Recall@10 | Routed | Gain on routed (NDCG) | Random, same rate | Coverage | Mean log-pop | Tail share | ms/case |
|---|---|---|---|---|---|---|---|---|---|
| always diffusion | 0.0402 [0.0396, 0.0409] | 0.0707 | 0.00 | – | – | 0.208 | 5.46 | 0.001 | 0.029 |
| always EASE | 0.0506 [0.0506, 0.0506] | 0.0862 | 1.00 | – | – | 0.629 | 5.22 | 0.030 | 0.009 |
| judge: controls + lists | 0.0506 [0.0506, 0.0506] | 0.0862 | 1.00 | +0.0104 | 0.0506 | 0.629 | 5.22 | 0.030 | 0.038 |
| judge: controls | 0.0506 [0.0506, 0.0506] | 0.0862 | 1.00 | +0.0104 | 0.0506 | 0.629 | 5.22 | 0.030 | 0.038 |
| entropy | 0.0506 [0.0506, 0.0506] | 0.0862 | 1.00 | +0.0104 | 0.0506 | 0.629 | 5.22 | 0.030 | 0.038 |

## Routing rate by history length (test, mean over runs)

| Policy | 1 | 2-3 | 4-9 | 10-29 | 30+ |
|---|---|---|---|---|---|
| judge: controls + lists | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| judge: controls | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| entropy | 1.00 | 1.00 | 1.00 | 1.00 | 0.99 |

Fallback cost: build 0.73 s, 0.0091 ms/case (p95 batch 0.0093), peak GPU memory 3.0 GB. Diffusion cost per case is the logged trajectory run (includes logging, an upper bound). The routed cost always includes the full diffusion run, because the risk needs the completed trajectory.

Quality-rate curves: `<dataset>_curve.csv` (policies, random, oracle).
