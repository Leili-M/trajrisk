# Uncertainty signals and controls

Every signal gives one number per evaluation case. All signals are oriented so that a
**higher value means a higher estimated risk** that DiffRec's final top-10 misses the
held-out item (proposal section 1: "Higher U indicates greater estimated risk").
An AUROC below 0.5 for a signal therefore means the data points the opposite way.

Notation follows the proposal: `s(s)` scores at denoising step s, `s*` final scores,
`L(s)`, `L*` the top-K lists after hiding already seen items, K = 10 unless stated.
DiffRec in all final models uses 5 steps and `noise_scale = 0.0001`.

## Label (never a predictor)

| Column | Definition | Proposal |
|---|---|---|
| `miss` | 1 if the held-out item is not in `L*` | F(u) = 1[target ∉ L*], formula 11 |
| `miss_one_sample` | same, for one random sample (repeated-sampling runs only) | — |

Outcome columns are listed in `risk_eval.LABEL_COLUMNS` and rejected as predictors
(`risk_eval.check_predictors`, tested in `tests/test_proposal_s1.py`).

## Trajectory signals (`src/trajrisk/stochastic.py`, `case_signals`)

Run with one clean trajectory: `--samples 1 --no-noisy-start --no-sampling-noise`.

| Column | Definition | Proposal |
|---|---|---|
| `step_change@10` | mean over the 4 non-final steps of `1 − |L(s) ∩ L*| / 10` | U_set, formula 4, uniform weights, K = 10 |
| `step_change@50` | same with K = 50 | U_set, K = 50 |
| `step_score_change` | `‖s(1) − s*‖₂ / ‖s*‖₂`, first (noisiest) step against the final scores | added (score-level instability) |

## Final-output confidence

| Column | Definition | Proposal |
|---|---|---|
| `small_margin` | `−(s*₍₁₀₎ − s*₍₁₁₎)`, the negated top-K boundary margin | m(u), formula 6 (negated so higher = riskier) |

## Masking signals (`src/trajrisk/masking.py`, `masked_signals`)

Hide each liked item with probability 0.5 (a history is never emptied), 10 masks per case.
The judged list is always `L*` from the full history.

| Column | Definition |
|---|---|
| `mask_change@10` | mean over masks of `1 − overlap(masked top-10, L*)` |
| `mask_change@50` | same with K = 50 |
| `mask_score_change` | mean over masks of `‖s_masked − s*‖₂ / ‖s*‖₂` |
| `mask_unstable@10` | share of `L*` items that drop out in at least one mask |

## Repeated sampling (`case_signals` with `--samples 10`, proposal section 7 baseline)

| Column | Definition |
|---|---|
| `sample_disagree@10`, `@50` | mean `1 − overlap` between each sample's top-K and the averaged top-K |
| `sample_rank_spread` | spread across samples of the ranks of the recommended items |

## Text signals (`scripts/semantic_experiment.py`, Amazon datasets)

Item vectors: all-MiniLM-L6-v2 embeddings of title and description, L2-normalised.
List meaning = normalised mean of its 10 item vectors.

| Column | Definition |
|---|---|
| `traj_count` | mean over the 5 steps of `1 − overlap(L(s), L*)` (final step adds 0) |
| `traj_sem` | mean over the 5 steps of `1 − cos(meaning(L(s)), meaning(L*))` |
| `mask_count`, `mask_sem` | same two measures under masking (50%, 10 masks) |
| `list_spread` | `1 −` mean pairwise cosine among the 10 recommended items |
| `hist_spread` | `1 − ‖mean history vector‖` |
| `hist_to_list` | `1 − cos(meaning(history), meaning(L*))` |

## Controls (proposal RQ1: final-score confidence, history length, popularity)

| Column | Definition |
|---|---|
| `log_history` | `log(1 + n_prior)`, number of earlier liked items |
| `small_margin` | see above |
| `history_popularity` | mean `log(1 + training positives)` of the user's earlier liked items |
