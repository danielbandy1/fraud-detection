# Full-source retrain — HELIOS, 2026-11-02

AUTHORED BY HELIOS. Branch `helios/full-retrain-20261102`.

This is the real, reproducible retrain behind the résumé line: the **full public
PaySim source** (already cached `data/raw/paysim_source.zip`, SHA
`f7eef9ff…596613d`, CC BY-SA 4.0), the repaired **leak-free** features, a
**time-ordered split**, and a **validation-only cost threshold**. Nothing is
downloaded; nothing is sampled; no random split is used.

## What was run

`full_retrain_20261102.py`, committed with this report. One command:

```bash
nice -n 15 bash -c 'ulimit -v 8000000; OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
MKL_NUM_THREADS=2 .venv/bin/python full_retrain_20261102.py'
```

Design, fixed before the run:

- **Population**: all 6,362,620 source rows; the pipeline's CASH_OUT/TRANSFER
  filter is kept (fraud only occurs there); exact-row deduplication.
  Eligible: 2,770,409 rows (8,213 fraud).
- **Split by time**: train steps ≤ 500 (2,645,779 rows, 5,561 fraud),
  validation 501–600 (81,080 rows, 1,052 fraud), test > 600 (43,550 rows,
  1,600 fraud, 3.67% fraud rate). Same-step records cannot see one another.
- **Features**: `src/features.build_features` — strictly earlier-step history
  counts for origin/destination accounts (ORION repair), balance deltas,
  drained/unchanged flags, log amount, hour/day/night. Account names are
  factorized to integer IDs. Features are float32 (LightGBM's native input).
- **Model**: LightGBM binary, the FORWARD_TIME_PLAN parameters verbatim
  (lr 0.05, ≤200 trees, 31 leaves, min_child_samples 50, feature/bagging 0.8,
  bagging_freq 5, class ratio from training only, seed 42, 2 threads,
  deterministic), early stopping 20 on validation AP.
- **Threshold**: `src.models.optimal_threshold` (the COST_THRESHOLD_FIX
  helper) fit on **validation labels only**, FN cost 1.0 / FP cost 0.1 — the
  repo's operating-cost basis (`src/models.py` `evaluate()`), with the
  no-alert operating point included and ties resolved to the lowest threshold:
  threshold = **0.995684**.
- **Test used only once**, after the threshold was frozen at 0.995684.

## Results — test block only (steps > 600)

| metric | value |
|---|---|
| average precision (AP) | **0.8802666172** |
| AUROC | **0.9955702175** |
| threshold (validation cost, FN=1/FP=0.1) | 0.9956841351 |
| precision at threshold | **0.8781163435** (TP 1585, FP 220, FN 15, TN 41730) |
| recall at threshold | **0.990625** |
| rows test | 43,550 |

Runtime **77.1 s**, peak RSS **≈1.9 GiB**, minimum system free memory stayed
above 2 GiB (asserted), nice 15, 2 threads, float32 features, 8 GiB
address-space cap. Full record with hashes, splits, resource trace, and saved
predictions/model: `results/full_retrain_20261102.json`
(`results/full_retrain_20261102_artifacts/` — validation/test prediction
parquet + model, kept local and git-ignored; metrics are reconstructable from
the parquet files).

## How this compares with the résumé's old 0.880 / 0.989

The old numbers (AP 0.880, AUROC 0.989) came from **random CV with leaky
full-window histories and a threshold tuned on evaluated labels** — ORION's
audit verdict stands: those are unsupported as evidence. The new run is honest
by construction, and:

- **AP 0.8803** — almost the same *number* as the old 0.880. That is a
  **coincidence, not corroboration**: different data split, different features,
  different threshold. Do not present the two as "the old claim is confirmed".
- At the cost threshold, **precision 87.8% / recall 98.8–99.1%** (test) vs the
  old artifact's precision 87.5% / recall 98.0% under confusion-count
  arithmetic — again similar-looking numbers from an entirely different design.
- Versus ORION's 2026-10-23/24 forward-time runs (AP 0.8895, AUROC 0.9946,
  float64 features, same split and parameters): this run's **AP is lower**
  (0.8803) and AUROC slightly higher (0.9956). The float32 feature cast required
  by shared-host limits is enough to change tree structure slightly. Report the
  numbers that this run actually produced; both runs are bounded evidence, not
  deployment validation.

Résumé sentence this supports: *"Trained a fraud-detection model on the full
public PaySim corpus (6.36 M rows) with leak-free prior-step account-history
features and a time-based train/validation/test split; held-out-test AP 0.880,
AUROC 0.996, and a cost-optimal threshold chosen on validation data reaching
88% precision at 99% recall."* It does **not** support the old random-CV 0.880/
0.989 claim, and JARVIS re-runs the numbers (see VERIFY below) before anything
goes in the résumé.

## How to re-verify

```bash
cd /home/daniel/nexus/projects/ds_portfolio/fraud-detection
python3 -c "import json;d=json.load(open('results/full_retrain_20261102.json'));\
assert all(k in d for k in ['ap','auroc','rows_test','split_steps']);print(d['ap'],d['auroc'],d['rows_test'])"
bash reproduce.sh   # 36-test suite incl. strict-prior-step and cost-threshold checks
```

Re-running `full_retrain_20261102.py` on this branch reproduces the metrics
(deterministic LightGBM, fixed seed; ±tiny AP jitter possible from binning).

## INTERVIEW_GUIDE

Plain-words explanations of every modelling choice, for Daniel to explain in
interviews.

**Why a time split instead of random CV.** A model must score transactions
*after* the ones it learned from. With random CV, almost every account in
validation/test also appears in training with its future — the model memorises
accounts, and scores flatter anything deployable. PaySim runs in one-hour
"steps", so I cut by step: hundreds of hours to train, ~100 for validation,
the last ~143 hours are the test block I report.

**Why "strictly prior-step" history features.** Counting, per account, "how
many transfers has chatter from this sender/destination sent before" is the
most tell-tale signal in PaySim — fraud drains an account. But if that count
sees the *future*, it is a leak: the feature is built from what has not
happened yet. My counts only look at rows from strictly earlier steps; rows
sharing a step cannot see each other. A regression test appends fake future
rows and checks past features do not change.

**Why CASH_OUT/TRANSFER only.** In PaySim every fraud label sits inside
CASH_OUT or TRANSFER; including other transaction types just inflates negatives
and muddies the base rate. I say this openly — it bounds what the score covers.

**Why LightGBM with those parameters.** Gradient-boosted trees are the
standard strong baseline on tabular data with mixed scales and nonlinear
interactions ("balance went to zero", "amount ≈ balance"). Parameters were
fixed *before* the run (the prespecified plan): learning rate 0.05, shallow
trees (31 leaves), large minimum leaf size to resist noise in a rare-fraud
setting, lightweight bagging/feature subsampling for stability, and class
weighting computed from the *training* block only. Early stopping watches
validation AP, never the test block.

**Why class weighting from training only.** Fraud is ~0.3% of eligible rows.
The weight handles imbalance; computing it on training alone avoids letting
later windows influence anything about the fit.

**How the threshold was chosen.** Precision/recall trade-offs are business
decisions. Here cost = 1·FN + 0.1·FP (the repo's standing cost basis): missing
a fraud is ten times worse than wasting an analyst's review. The helper
evaluates every score boundary — including *never alert* — on **validation**
labels and picks the cheapest. Only then is the threshold frozen and applied to
test. That is why the test "confusion matrix at 0.9957" is honest: the number
was never chosen to look good on the data it is evaluated on.

**What this does and does not prove.**
- *Does*: on the last ~143 simulated hours, the model ranks frauded
  transactions so that AP ≈ 0.88; alerting at the chosen threshold removes
  ~99% of frauds (1585 of 1600) with ~88% of alerts correct (220 false
  alarms in 43,550 transactions).
- *Does not*: real-world transferability — PaySim is synthetic, the test
  window was already scored once in bounded earlier evaluations (I say so
  plainly), and balance-after-transfer features mean this is *detection after
  the transaction posts*, not blocking before it.

**Likely interviewer follow-ups.** "Isn't 0.99 AUROC just class imbalance?" —
No: AUROC is prevalence-robust; class imbalance is why we lead with AP
(0.880) and precision/recall at an operational threshold. "Is history available
online?" — No, single-record scoring needs a stateful history store; batch is
educational. "What did you tune?" — Nothing on test; every choice was
prespecified or validation-based; the honest statement is the constraint,
not a leaderboard.
