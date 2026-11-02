# Pre-authorisation fraud model — HELIOS, 2026-11-02

AUTHORED BY HELIOS. Branch `helios/full-retrain-20261102`.

Follow-up to [`RESULTS_FULL_RETRAIN.md`](RESULTS_FULL_RETRAIN.md) implementing
point 2 of [`JARVIS_REVIEW_FULL_RETRAIN_20261102.md`](JARVIS_REVIEW_FULL_RETRAIN_20261102.md):
a bank deciding whether to **approve** a transaction does not yet see balances
*after* it. The identical pipeline is retrained with every post-transaction
field removed.

## What was run

`full_preauth_20261102.py` (committed with this report); same command shape as
before:

```bash
nice -n 15 bash -c 'ulimit -v 8000000; OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
MKL_NUM_THREADS=2 .venv/bin/python full_preauth_20261102.py'
```

Identical design to the post-transaction run — same cached source ZIP (hash
checked, never re-downloaded), same time split (≤500 / 501–600 / >600), same
seed 42 LightGBM parameters, same class weighting from training only, same
validation-only cost threshold (FN = 1.0, FP = 0.1, `src.models.optimal_threshold`,
frozen before any test prediction), float32 features, 2 threads, memory floor
2 GiB monitored. Excluded features: `newbalanceorig`, `newbalancedest`,
`orig_balance_delta`, `dest_balance_delta`, `orig_drained`, `dest_no_change`.
Kept (13): `amount`, `log_amount`, `oldbalanceorg`, `oldbalancedest`,
`amount_to_orig_balance`, `hour`, `day`, `is_night`, `orig_tx_count`,
`orig_unique_dest`, `orig_step_gap`, `dest_recv_count`, `is_transfer`.

## Results — test block only (steps > 600, 43,550 rows, 1,600 fraud, 3.67%)

| metric | post-transaction model | pre-authorisation model |
|---|---|---|
| test fraud base rate | 3.67% | 3.67% (same split) |
| average precision (AP) | **0.8802666172** | **0.8366078463** |
| AUROC | **0.9955702175** | **0.9896117104** |
| threshold (validation cost, FN=1/FP=0.1) | 0.9956841351 | 0.9999999832 |
| precision at threshold | 0.8781163435 (FP 220) | 0.8333333333 (FP 317) |
| recall at threshold | 0.990625 (FN 15) | 0.990625 (FN 15) |
| confusion TP/FP/FN/TN | 1585 / 220 / 15 / 41730 | 1585 / 317 / 15 / 41633 |
| best iteration | 17 | 6 |

Runtime 75.0 s, peak RSS 1.92 GiB, minimum system free memory 5.35 GiB (floor
2 GiB respected). Full record: `results/preauth_20261102.json`; test
predictions + model kept local and git-ignored in
`results/preauth_20261102_artifacts/` for independent re-verification.

## Three plain sentences on what changed and why

1. The retrain is identical to the first one except that every feature reading
   a post-transaction balance is dropped — `newbalanceorig`,
   `newbalancedest`, both balance deltas, and the drained/unchanged-balance
   flags — leaving only what a bank knows before approval: amounts, the
   pre-existing balances, the amount-to-balance ratio, time features, the
   strictly prior-step account-history counts, and the transaction type.
2. On the same held-out test block at the same 3.67% fraud rate, AP fell from
   0.880 to 0.837 and AUROC from 0.9956 to 0.9896, and the cost-optimal
   operating point moved to alerting only with near-certain scores (precision
   83.3%, recall 99.1% over 43,550 rows) — the model lost the shortcut of
   reading the transaction's own outcome from the balances.
3. This is the honest number for any "block before it posts" claim: part of
   the post-transaction model's higher skill was learning PaySim's simulator
   pattern (fraud zeroes the account, destination balances never change);
   résumé and interview talk should quote AP 0.837 at 3.7% prevalence for
   pre-authorisation and AP 0.880 only for post-transaction scoring. AP is not
   comparable with the historical 0.880/0.989 random-CV artifact, which
   remains unsupported as evidence either way.

## VERIFY

```bash
cd /home/daniel/nexus/projects/ds_portfolio/fraud-detection
python3 -c "import json;d=json.load(open('results/preauth_20261102.json'));\
assert not any('newbalance' in f or 'delta' in f or 'drained' in f or 'no_change' in f \
for f in d['features']);print(d['ap'],d['auroc'])"
```

`bash reproduce.sh` still passes (58 tests, exit 0; unchanged suite).
