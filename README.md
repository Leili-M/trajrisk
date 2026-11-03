# trajrisk — Trajectory-Based Uncertainty for Diffusion Recommenders

NLP course project, University of Oulu (Leili Motahari). The report is organised by the
sections of the research proposal; commits are prefixed `[P1]` … `[P10]` with the same
numbers, so each part of the proposal can be reviewed on its own.

**Question.** Do the intermediate predictions of one diffusion sampling run (the denoising
trajectory) tell when a recommendation is wrong — beyond final-score confidence, history
length and item popularity? And does comparing trajectories by item attributes, instead of
item identities, separate suitable alternatives from real failures?

## Repository layout

| Path | Content |
|---|---|
| `configs/` | one frozen configuration per dataset (split dates, catalog rule, model settings) |
| `src/trajrisk/` | library code (data, models, logging, signals, evaluation) |
| `scripts/` | one script per experiment; each writes a report under `reports/` |
| `tests/` | unit tests (`PYTHONPATH=src python -m pytest -q tests`) |
| `reports/` | all result tables (Markdown + JSON), committed with the code |
| `docs/` | signal definitions (`signals.md`), human-audit guide (`audit_guide.md`) |
| `jobs/run.sh` | Slurm wrapper used on CSC Roihu (requests one GPU) |
| `external/` | DiffRec (official, commit `e42532f`); GiffCF authors' code (only its data split is used) |

No data are included (licences of HUMMUS, MovieLens and Amazon Reviews 2023 do not allow
redistribution); the download scripts fetch them and store checksums.

## Environment

CSC Roihu (NVIDIA GH200), module `python-pytorch/2.13`:

```bash
module load python-pytorch/2.13
cd /projappl/project_2021634/trajrisk
PYTHONPATH=src python -m pytest -q tests
```

Every experiment below is submitted as
`sbatch --partition=gpumedium --time=HH:MM:SS jobs/run.sh <config> <script> <arguments>`.
For readability the commands below show only `python <script> <arguments>`; scripts that
take `--config` read all paths and settings from it. Options of every script:
`python scripts/<script>.py --help` and the script's docstring.

Datasets and model runs:

| Config | Dataset | DiffRec runs (5 seeds) | GiffCF runs (3 seeds) | EASE λ |
|---|---|---|---|---|
| `configs/movielens.yaml` | MovieLens-1M | ml_base, ml_seed1–4 | – | 500 |
| `configs/amazon_video_games.yaml` | Amazon Video Games | vg_base, vg_seed1–4 | vg_giff_s1–s3 | 50 |
| `configs/amazon_books.yaml` | Amazon Books | bk_seed42, bk_seed1–4 | bk_giff_s1–s3 | 20 |
| `configs/base.yaml` | HUMMUS | hm_seed42, hm_seed1–4 | – | 100000 |

## Reproducing the report, section by section

Rule used everywhere: every choice (model settings, weights, temperatures, thresholds,
reranker coefficients) is made on validation data; the test cases are evaluated once.
Target ranks are stored in separate label files that the signal code never reads.

### P1 — Prediction task and leakage checks (report: Prediction task, Preventing information leakage)
- `scripts/check_proposal_s1.py` → `reports/checks/proposal_s1.md` (checks on real data)
- `tests/test_proposal_s1.py` (unit tests: no future events, no seen items, label guard)

### P2 — Data and temporal protocol (Datasets, Temporal split, Attrition; Tables cases, datasets, split, attrition)
- `scripts/download_hummus.py`, `download_movielens.py`, `download_amazon.py`
- `scripts/prepare_data.py` (positive interactions, frozen split, catalog, cases)
- `scripts/audit_data_s2.py` → `reports/checks/proposal_s2.md`

### P3 — Controlled recipe benchmark (Controlled benchmark; Tables recipe_audit, scenarios, parsing)
- `scripts/benchmark_attributes.py` → `reports/benchmark/attributes.md`
- `scripts/build_scenarios.py --config configs/base.yaml --n-valid 200 --n-test 400`
  → `reports/benchmark/scenarios.md`
- `scripts/render_requests.py`, `scripts/llm_requests.py` (templates, Qwen2.5-7B paraphrases, parsers)
- `scripts/audit_sheet.py` (human audit sheet and its summary)

### P4 — Backbones and trajectory logging (Recommender and trajectory logging; Tables baselines, steps, giffcf_repro, giffcf)
- `scripts/train_diffrec.py` (DiffRec runs)
- `scripts/tuned_baselines.py` → `reports/baselines/<dataset>.md` (popularity, recent popularity,
  conditional probability, ItemKNN, EASE; grid on model-validation, test once)
- `scripts/log_trajectory.py --config <c> --runs <runs>` → `<output>/trajlogs/<run>__<split>/`
  (top-50 lists and scores per checkpoint, noise levels, separate `labels.parquet`, manifest
  with hashes) and `reports/trajlogs/<dataset>[_giffcf].csv`
- `scripts/step_count_experiment.py` → `reports/step_counts/<dataset>.md`
- GiffCF (own PyTorch re-implementation): `scripts/giffcf_reproduce.py`
  → `reports/backbones/giffcf_reproduction.md`; `scripts/train_giffcf.py`;
  `scripts/eval_backbone.py` → `reports/backbones/<dataset>.md`

### P5 — Uncertainty signals (Uncertainty signals; Table signals)
- `scripts/compute_signals.py --config <c> --runs <runs>` → `signals.parquet` next to each log,
  `reports/signals/<dataset>[_giffcf].md` (single-signal AUROC on risk-validation, choice of
  weighting ρ and entropy temperature τ)

### P6 — Preference-aware comparison (Preference-aware comparison; RQ2)
- `scripts/preference_experiment.py --config configs/base.yaml --runs hm_seed42 hm_seed1 hm_seed2 hm_seed3 hm_seed4`
  → `reports/preference/hummus.md` (reranker β, γ chosen on validation scenarios; soft and
  hard pipelines; attribute and utility distances with Hungarian matching; false alarms at
  matched detection; diagnostic subsets)

### P7 — Risk model and baselines (Risk model and baselines; Tables rq1, books_detail)
- `scripts/perturbation_signals.py --config <c> --runs <DiffRec runs>` (repeated sampling 2/5/10
  and masked histories; must run before the judge to be included)
- `scripts/judge.py --config <c> --runs <runs>` → `reports/judge/<dataset>[_giffcf].md`
  (logistic regression fitted on risk-validation, evaluated once on test; controls,
  trajectory, list-only ablation, sampling and masking sets; second protocol with
  cross-validation inside test for comparison with the first experiments)

### P8 — Evaluation (Evaluation; Table subgroups)
- AUROC, AUPRC with prevalence, Brier score, risk–coverage area (miss and 1−NDCG@10, random and
  oracle references), reliability tables: written by `scripts/judge.py`
- `scripts/subgroups.py --config <c> --runs <runs>` → `reports/subgroups/<dataset>[_giffcf].md`
  (history length and target popularity groups fixed in advance; paired user bootstrap)
- `scripts/make_figures.py` → `reports/figures/risk_coverage.pdf`, `reliability.pdf`
  (from the files written by `judge.py`; the judge also saves its fitted coefficients in
  `reports/judge/<dataset>[_giffcf]_parameters.json`)

### P9 — Uncertainty-guided fallback (Uncertainty-guided fallback; RQ4)
- `scripts/fallback_ranks.py --config <c> --lam <λ>` (frozen EASE on risk-validation and test;
  time and memory)
- `scripts/routing.py --config <c> --runs <runs>` → `reports/routing/<dataset>[_giffcf].md`
  and `_curve.csv` (threshold chosen on risk-validation; random routing at the same rate;
  coverage, popularity, routing rate by history length, cost per request)

### Order of a full run for one dataset

```text
prepare_data → train_diffrec (5 seeds) → tuned_baselines → log_trajectory
→ compute_signals → perturbation_signals → judge → subgroups
→ fallback_ranks → routing
```

## Earlier exploratory scripts

`trajectory_experiment.py`, `masking_experiment.py`, `semantic_experiment.py`,
`semantic_vs_trajectory.py`, `routing_experiment.py`, `truncation_experiment.py`,
`run_baselines.py`, `seed_summary.py`, `control_check.py` and related helpers belong to the
first round of experiments, before the proposal-by-section implementation. They are kept for
transparency; the final tables of the report come only from the scripts listed above.

## Licences and credits

HUMMUS (educational and scientific use), MovieLens-1M (research use, cite Harper & Konstan
2015, no redistribution), Amazon Reviews 2023 (research use). DiffRec: official code of
Wang et al. (SIGIR 2023). GiffCF: re-implemented from Zhu et al. (SIGIR 2024); the authors'
repository has no licence file and its code is not used. Qwen2.5-7B-Instruct for paraphrases
and parsing.
