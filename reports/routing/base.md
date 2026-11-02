# Section 9: routing to EASE, base (DiffRec)

Threshold chosen on risk-validation (cross-fitted judge), applied once to test. EASE lambda 100000 (section 4). Mean over runs [95% t-interval]. Quality is over ALL test cases.

| Policy | NDCG@10 | Recall@10 | Routed | Gain on routed (NDCG) | Random, same rate | Coverage | Mean log-pop | Tail share | ms/case |
|---|---|---|---|---|---|---|---|---|---|
| always diffusion | 0.0081 [0.0077, 0.0086] | 0.0157 | 0.00 | – | – | 0.182 | 5.88 | 0.014 | 0.165 |
| always EASE | 0.0084 [0.0084, 0.0084] | 0.0162 | 1.00 | – | – | 0.318 | 6.33 | 0.012 | 0.117 |
| judge: controls + lists | 0.0084 [0.0081, 0.0087] | 0.0161 | 0.76 | +0.0004 | 0.0084 | 0.279 | 6.32 | 0.010 | 0.253 |
| judge: controls | 0.0084 [0.0081, 0.0086] | 0.0162 | 0.84 | +0.0003 | 0.0084 | 0.285 | 6.34 | 0.010 | 0.262 |
| entropy | 0.0084 [0.0084, 0.0085] | 0.0162 | 0.68 | +0.0004 | 0.0084 | 0.321 | 6.14 | 0.016 | 0.244 |

## Routing rate by history length (test, mean over runs)

| Policy | 1 | 2-3 | 4-9 | 10-29 | 30+ |
|---|---|---|---|---|---|
| judge: controls + lists | 0.64 | 0.64 | 0.65 | 0.67 | 0.85 |
| judge: controls | 0.64 | 0.68 | 0.73 | 0.79 | 0.94 |
| entropy | 1.00 | 1.00 | 0.97 | 0.84 | 0.45 |

Fallback cost: build 8.41 s, 0.1169 ms/case (p95 batch 0.2349), peak GPU memory 30.35 GB. Diffusion cost per case is the logged trajectory run (includes logging, an upper bound). The routed cost always includes the full diffusion run, because the risk needs the completed trajectory.

Quality-rate curves: `<dataset>_curve.csv` (policies, random, oracle).
