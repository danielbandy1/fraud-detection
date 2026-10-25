# Full PaySim evaluation — time-ordered, uncertainty and validation threshold

AUTHORED BY ORION

**Share with caveats.** Full public source reprocessed on 2026-10-24T23:22:03-0400; no sampling. Fixed leakage-repaired histories/model/split from23Oct, now with a validation-F1 operating threshold and bootstrap uncertainty. This is a re-evaluation of a **previously seen test window**, not a newly untouched holdout or independent replication. No claim of production performance or model superiority to the historical random-CV headline.

## Main results

| Metric | Test estimate | Primary95%bootstrapCI | Definition |
|---|---:|---|---|
| Average precision (AP) | 0.889491591274 | [0.818801752535, 0.946536193696] | sklearn average_precision_score; not trapezoidal PR area |
| AUROC | 0.994632389750 | [0.991558713008, 0.997160090524] | Rank-based ROC area with tied-score handling |
| Fraud base rate | 0.036739380023 (3.673938%) | — | 1600/43550 |
| Precision | 0.885714285714 | — | TP/(TP+FP) at validation-selected threshold |
| Recall | 0.988125000000 | — | TP/(TP+FN) |
| F1 | 0.934121122600 | — | 2TP/(2TP+FP+FN) |

Threshold **0.9974552615977691**, chosen by maximum validation F1 (exact ties choose the higher threshold), validation F1=0.855030800821. The rule was written in `FULL_EVAL_PLAN_ORION_20261024.md` before this run. Threshold fixed before test prediction in the executable path; test data do not enter fitting, early stopping or threshold selection. Same validation window selects the early-stopping iteration and threshold, so validation performance is optimistic. Model uses training-only class weighting; probabilities are not calibrated fraud risks, and the high threshold is not a business-loss optimum.

Confusion matrix (rows actual0/1; columns predicted0/1):

| Actual / predicted | 0 | 1 |
|---|---:|---:|
| 0 | 41746 | 204 |
| 1 | 19 | 1581 |

At the original fixed0.5 comparator: TN=41689,FP=261,FN=19,TP=1581; precision=0.858306188925,recall=0.988125000000. This comparator is reported, not used to pick a threshold on test results.

## Data and leakage boundary

Public source [ealaxi/paysim1](https://www.kaggle.com/datasets/ealaxi/paysim1), CC BY-SA4.0 as recorded in `docs/source_metadata.json`. ZIP SHA256 `f7eef9ffad5cfa64a034143a5c9b30491d189420b273d5ad5723ca40b596613d`. Raw 6,362,620transactions/8,213fraud; the pipeline’s declared population retains all 2,770,409 CASH_OUT/TRANSFER rows (all8213frauds), exact duplicates removed=0. Other transaction types are outside the analysis population: “full” means whole source read and all eligible rows used, not all types modeled.

| Window | Step range | Rows | Frauds | Prevalence |
|---|---|---:|---:|---:|
| train | 1–500 | 2645779 | 5561 | 0.002101838438 |
| validation | 501–600 | 81080 | 1052 | 0.012974839665 |
| test | 601–743 | 43550 | 1600 | 0.036739380023 |

19features; original account strings are equivalence-preservingly integerized, not treated as numerical predictors. Histories use strictly earlier steps, never labels, future rows or same-step observations. Later windows can use previously observed unlabeled transaction histories, as in sequential operation. Balances after transaction remain inputs, so this evaluates **post-transaction simulated detection**, not pre-payment intervention. No real-world or held-out-customer validation is claimed.

LightGBM settings fixed: learning rate.05,31leaves,min_child_samples50,feature/bagging.8,bagging_freq5,seed42,two threads,200maxrounds,20round validation-AP early stopping. Chosen iteration=18. No new hyperparameter search; configuration and source/code hashes are saved in `FULL_EVAL_ORION_20261024.json`.

## Bootstrap design and limits

Primary: 2000circular moving-block bootstrap draws of24consecutive hourly steps, concatenated/truncated to the143-hour test timeline, resampling complete step clusters; seed20261024. All2000draws valid; one-class draws=0. Score-bin weighted AP/AUROC retains tie handling; ten actual bootstrap draws cross-checked against sklearn using transaction weights.

This retains within-step dependence and local day-length patterns, but assumes approximate temporal stationarity over a short test window. Circular wrapping connects the final and initial test steps. Account dependence beyond a24-hour block and future shifts remain unaddressed. It is a **conditional fixed-model interval**: no model refitting/selection uncertainty, no population guarantee beyond these observed simulated steps.

Sensitivity only:2000class-stratified iid-row draws,seed20261025 (ignores time/account dependence) give AP95%CI[0.873604439762,0.904756748886],AUROC95%CI[0.991606065256,0.997004741322]. The much narrower AP interval illustrates why iid rows are not the primary uncertainty calculation. Class totals are fixed in that sensitivity, whereas time-block resampling allows prevalence to vary.

## Historical headline comparison

| Design | AP | AUROC | Status |
|---|---:|---:|---|
| Historical random-CV artifact | 0.879995249726 | 0.989433941805 | Full-window history leakage; threshold chosen on evaluated labels; not endorsed |
| Repaired forward-time test | 0.889491591274 | 0.994632389750 | Known holdout; strict-prior histories, validation-only selection; conditional uncertainty above |

Old eligible-sample prevalence8213/2770409≈0.2965% vs forward test3.6739%; AP is prevalence-sensitive. Differences are not an apples-to-apples improvement estimate. Legacy0.880/0.989 are preserved for traceability, not reinstated as résumé claims.

## Reproduce and verify

From this repository:

```bash
nice -n 15 .venv/bin/python full_eval.py
.venv/bin/python verify_full_eval.py
nice -n 15 bash reproduce.sh
```

The first command re-reads the pinned source and writes `full_eval_results/{validation,test}_predictions.parquet`, `model.txt` and the metrics JSON. It requires the already-downloaded public source ZIP and pinned `.venv`; no network in evaluation. The metrics JSON governs all numbers here. Re-running intentionally refreshes these generated evaluation files; preserve them first if retaining the exact run is needed.

Actual nice=15,two threads,peakRSS=1918.515625MiB,minimumMemAvailable=5.932938GiB,263checks at0.2seconds; elapsed=55.296167s. Worker has6GiB address-space limit, aborts if MemAvailable<3GiB or peakRSS>=6GiB. Observed RSS stayed below2GiB (well below6GB); no stop condition hit. NoMCC.

Verification: `verify_full_eval.py` exited0: independently reconstructed validation threshold from ranked predictions, AP via tied precision sums, AUROC via average ranks, both confusion matrices, split boundaries, file hashes and resource evidence. `nice -n15 bash reproduce.sh` exited0,**23tests passed** (3.65s). `reproduce.sh` is the existing bounded fixture/provenance/test command; it does not itself retrain PaySim—the full-run command above does. Existing audit.json was backed up and byte-verified before that command rewrote its generated copy.

No push, commit, résumé edit, public-page edit, deployment or external submission. New files are local review artifacts. Assessment: useful reproducible simulation evaluation with explicit conditional uncertainty; insufficient by itself for deployment or a causal performance-improvement claim.
