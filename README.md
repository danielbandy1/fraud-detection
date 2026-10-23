# PaySim fraud detection — historical result audit and repaired histories

AUTHORED BY ORION — local repairs for Daniel's review, 23 October 2026.

**Question:** can transaction features identify rare simulated fraud without looking into the future? **Update: a new full-source forward-time evaluation is now available; still not deployment-ready.**

See [the forward-time report](FORWARD_TIME_EVAL_ORION_20261023.md) and adjacent JSON: later-step AP 0.889492 / AUROC 0.994632, test fraud rate 3.673938%, threshold0.5, precision85.8306% / recall98.8125%. The new design uses all source data, train<=500, validation501–600, test>600, and strictly earlier-hour histories. It does not retrospectively validate the historical metrics below. Its exact reproduction command is in that report; the default `reproduce.sh` remains the lightweight historical-artifact/unit-test check.

```bash
bash reproduce.sh
```

The default run creates a clean Python 3.13 environment, installs exact pins, checks historical metric arithmetic, exercises strict prior-step histories and runs the unit suite. It uses explicitly labelled synthetic edge fixtures; **it does not train or evaluate on PaySim** and does not reproduce AUPRC/AUROC from predictions. No credentials, external service, publication or private data are used.

## Public data source / licence

PaySim is a **synthetic financial simulation**, not real bank-client transactions: https://www.kaggle.com/datasets/ealaxi/paysim1 . The public Kaggle dataset-list API entry `ealaxi/paysim1` reports **CC BY-SA 4.0**; code and data permissions are separate. No raw PaySim rows are redistributed by this branch. A full rerun needs the exact public source file under `data/raw/`, its independently recorded hash, explicit inclusion rules and a prespecified temporal evaluation. Original code's filtering retains CASH_OUT/TRANSFER, so results do not cover other transaction types. The old source's 6,362,620 total rows were not independently recounted here.

## What the saved artifact supports

`models/results.json` records **AUPRC 0.879995 / AUROC 0.989434**, 2,770,409 filtered rows, 8,213 fraud labels, and TP/FP/FN/TN counts. Those entries trace the résumé's rounded **0.880 / 0.989**, but the predictions/raw fingerprint are absent, so independent score reproduction is unavailable. Confusion counts reconcile to rows and fraud counts; `portfolio_results/audit.json` records that limited check and the artifact hash. The existing figure files are historical, not newly validated figures.

## Defects found and repaired

- Full-window account counts/unique destinations previously used future transactions before random CV. Histories now use **strictly earlier steps**; tied-hour records cannot see one another. Original row/index order is preserved so returned predictions stay aligned. Regression tests append future records and verify past features cannot change.
- Original README confused all-negative **accuracy** with AUROC. All-negative predictions can have high accuracy with rare fraud; a constant score has AUROC 0.5 when both classes exist. That claim is removed.
- Historical random-fold metrics and threshold tuning on evaluated labels are exploratory. `train.py` now requires an explicit exploratory-CV acknowledgement, and the repaired history definition invalidates reuse of previous model parameters/results as though unchanged.

## What is still needed

A later, prespecified forward-time train/validation/test run must use training-window-only state, tune thresholds on validation data and report held-out average precision, recall, review burden and cost. Balance-after-transfer features are available only after the transfer completes: this cannot honestly claim pre-transaction prevention. Single-record scoring has no account history unless supplied through a stateful history process; the batch helper is a retrospective educational interface, not a deployed online feature store. SHAP code/UI exist, but neither proves model validity. New deployment-performance numbers remain **unsupported**, not silently replaced with smoke-test results.

Old README preserved in `docs/README_UPSTREAM_20261023.md`. Python/ML/API/Streamlit code is inspectable, but no claim of a live production service or Daniel's independent authorship of these repairs is made.
