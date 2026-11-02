# Section 9: routing to EASE, movielens (DiffRec)

Threshold chosen on risk-validation (cross-fitted judge), applied once to test. EASE lambda 500 (section 4). Mean over runs [95% t-interval]. Quality is over ALL test cases.

| Policy | NDCG@10 | Recall@10 | Routed | Gain on routed (NDCG) | Random, same rate | Coverage | Mean log-pop | Tail share | ms/case |
|---|---|---|---|---|---|---|---|---|---|
| always diffusion | 0.0467 [0.0465, 0.0468] | 0.0946 | 0.00 | – | – | 0.338 | 6.62 | 0.000 | 0.035 |
| always EASE | 0.0498 [0.0498, 0.0498] | 0.1019 | 1.00 | – | – | 0.376 | 6.67 | 0.000 | 0.001 |
| judge: controls + lists | 0.0501 [0.0499, 0.0504] | 0.1025 | 0.97 | +0.0036 | 0.0497 | 0.376 | 6.68 | 0.000 | 0.036 |
| judge: controls | 0.0501 [0.0499, 0.0504] | 0.1025 | 0.97 | +0.0036 | 0.0497 | 0.376 | 6.68 | 0.000 | 0.036 |
| entropy | 0.0497 [0.0496, 0.0498] | 0.1018 | 0.98 | +0.0031 | 0.0497 | 0.377 | 6.67 | 0.000 | 0.036 |

## Routing rate by history length (test, mean over runs)

| Policy | 1 | 2-3 | 4-9 | 10-29 | 30+ |
|---|---|---|---|---|---|
| judge: controls + lists | 0.42 | 0.53 | 0.71 | 0.92 | 1.00 |
| judge: controls | 0.42 | 0.51 | 0.70 | 0.92 | 1.00 |
| entropy | 1.00 | 1.00 | 1.00 | 1.00 | 0.98 |

Fallback cost: build 0.49 s, 0.001 ms/case (p95 batch 0.001), peak GPU memory 0.17 GB. Diffusion cost per case is the logged trajectory run (includes logging, an upper bound). The routed cost always includes the full diffusion run, because the risk needs the completed trajectory.

Quality-rate curves: `<dataset>_curve.csv` (policies, random, oracle).
