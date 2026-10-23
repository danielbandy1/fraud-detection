# PaySim forward-time evaluation — 23 October 2026

AUTHORED BY ORION

**Full public source used; no subsampling.** Fixed plan recorded before the source was downloaded/evaluated. This is post-transaction detection in a synthetic simulation, not evidence of pre-payment prevention or production-bank performance.

## Results

| Metric | New later-step test | Historical random-CV artifact |
|---|---:|---:|
| Average precision (AUPRC) | 0.8894915913 | 0.8799952497 |
| AUROC | 0.9946323897 | 0.9894339418 |
| Positive base rate | 3.6739380023% | 0.2964544224% |
| Threshold | 0.5, fixed before evaluation | 0.9419679616, selected on evaluated labels |
| Precision at stated threshold | 85.8306188925% | 62.9487078784% |
| Recall at stated threshold | 98.8125000000% | 97.5770120540% |
| Evaluated rows | 43,550 | 2,770,409 |
| TP / FP / FN / TN | 1581 / 261 / 19 / 41689 | 8014 / 4717 / 199 / 2757479 |

**Do not call this an improvement over the old score.** Populations, validation design, prevalence and thresholds differ. The new test base rate is about 12.4 times the full filtered-source rate, and average precision depends on prevalence. Historical 0.880/0.989 numbers are retained solely for comparison and traceability; their future-history/random-CV defects are not repaired retrospectively.

## Population and fixed split

Source has **6,362,620 rows**, steps 1–743, and 8,213 fraud labels. CASH_OUT/TRANSFER inclusion retains **2,770,409 rows**; 0 exact duplicates removed. No whole steps were dropped or sampled. This remains a selected-transaction-type population, not all five transaction types.

| Window | Fixed step rule | Rows | Fraud | Base rate |
|---|---|---:|---:|---:|
| train | ≤500 (observed 1–500) | 2,645,779 | 5,561 | 0.210184% |
| validation | 501–600 (observed 501–600) | 81,080 | 1,052 | 1.297484% |
| test | >600 (observed 601–743) | 43,550 | 1,600 | 3.673938% |

## Method and leakage controls

- Feature histories use only strictly earlier steps; same-step records cannot see each other. Account IDs are integerized as equivalence classes for memory use and are not predictors. The feature implementation was vectorized with unchanged prior-step semantics, tested against an independent row-wise reference, future-row invariance and same-step isolation.
- Histories can accumulate earlier **unlabelled** observations through validation/test, as an online state would; no future rows or fraud labels enter them. This is explicitly different from freezing all history at training cutoff.
- LightGBM configuration was fixed in `FORWARD_TIME_PLAN_ORION_20261023.md`: at most 200 trees, 31 leaves,learning rate .05,two threads,training-only class weight,seed 42. Validation AP alone selected iteration 18 with patience 20. No test arrays are supplied to model fitting. No test-based hyperparameter or threshold search.
- Threshold **0.5** was fixed before the run. The labels in the latest time window are used for the final report only; the same predictions are subsequently read for independent verification, without retraining or selection.
- AUPRC means `average_precision_score`, not trapezoidal PR integration. The verifier independently reconstructs average precision by tied-score groups and AUROC using rank sums, plus confusion counts/precision/recall.
- All money/balance values retain float64 precision. Source receipt, script, plan, feature and prediction hashes are recorded in the adjacent JSON.

## Reproduce every metric

From this repository (initial environment installation may need internet; source download is public/anonymous):

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-portfolio.lock
nice -n 10 .venv/bin/python download_paysim_public.py
nice -n 10 .venv/bin/python forward_time_eval.py
nice -n 10 .venv/bin/python verify_forward_time.py
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python -m pytest -q tests
```

For a repeat without network, retain `data/raw/paysim_source.zip`. The training script refuses a different archive hash. Public source: https://www.kaggle.com/datasets/ealaxi/paysim1 ; anonymous download URL: https://www.kaggle.com/api/v1/datasets/download/ealaxi/paysim1 . Source ZIP SHA256: **f7eef9ffad5cfa64a034143a5c9b30491d189420b273d5ad5723ca40b596613d**. Kaggle metadata reports **CC BY-SA 4.0**; source is synthetic. Raw source rows and model/prediction caches remain local, Git-ignored. Derived result JSON/attribution can be reviewed before any future publication.

Adjacent machine-readable result: `FORWARD_TIME_EVAL_ORION_20261023.json`. Reusable held-out predictions: `forward_time_results/test_predictions.parquet`; trained18-iteration model: `forward_time_results/model.txt`. Download receipt: `data/raw/paysim_download_receipt.json`. Independent verification output: `../audit_evidence_20261023/forward_time_verify.json`.

## Resource record

Nice level **10**; two threads; address space capped at 6 GiB (stricter than a 6 GiB resident-memory ceiling). Measured peak RSS **1936.703 MiB**, elapsed **54.099s** excluding download/dependency installation. The full source fit comfortably, so no time-step subsample was necessary. This is the recorded run, not a timing guarantee on other machines.

## Proposed résumé bullet

**PaySim fraud detection (synthetic)** — LightGBM with strict prior-hour histories; forward-time holdout of 43,550 transactions (3.67% fraud): AUPRC 0.889, AUROC 0.995; 85.8% precision and 98.8% recall at prespecified threshold 0.5.

These numbers belong to this new evaluation, not the old 0.880/0.989 pipeline. Daniel must reproduce/explain the AI-assisted changes before adopting the bullet. One temporal split in a simulation does not establish robustness across real banks, customers, time regimes or business loss assumptions. Post-transfer balances make this an after-transaction detection task. No push, account creation, deployment, external submission or real-money action was performed.
