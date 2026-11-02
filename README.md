# PaySim fraud detection — historical result audit and repaired histories

AUTHORED BY ORION — local repairs for Daniel's review, 23 October 2026.

**Question:** can transaction features identify rare simulated fraud without looking into the future? **Update: a new full-source forward-time evaluation is now available; still not deployment-ready.**

See [the full forward-time evaluation report](FULL_EVAL_ORION_20261024.md) and [earlier memo](FORWARD_TIME_EVAL_ORION_20261023.md): later-step test AP **0.889492** (95% bootstrap CI: [0.8188, 0.9465]) and AUROC **0.994632** (95% CI: [0.9916, 0.9972]) on 43,550 held-out transactions (3.67% fraud rate).
- At validation-tuned F1 threshold (0.9975): Precision = **88.57%**, Recall = **98.81%**, F1 = **0.9341** (TP: 1581, FP: 204, FN: 19).
- At default threshold (0.50): Precision = **85.83%**, Recall = **98.81%** (TP: 1581, FP: 261, FN: 19).
- Holdout curves plotted in `portfolio_results/forward_eval_curves.png`.
The new design uses all source data, train<=500, validation 501–600, test>600, and strictly earlier-hour histories. It does not retrospectively validate the historical metrics below. Its exact reproduction command is in that report; the default `reproduce.sh` remains the lightweight historical-artifact/unit-test check.

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

## Full-source retrain (HELIOS branch, 2026-11-02)

A real retrain on the full cached public source with leak-free prior-step histories and a time-ordered train/validation/test split (steps <=500 / 501-600 / >600, train-only class weighting), LightGBM plan parameters fixed in advance, and a cost-based threshold (FN=1/FP=0.1) fit on validation labels only. Held-out **test block** (43,550 rows, 1,600 fraud): AP **0.8802666172**, AUROC **0.9955702175**, precision **0.8781163435** and recall **0.990625** at threshold 0.995684 (TP 1585 / FP 220 / FN 15 / TN 41730). Full record and reproduction: [results/full_retrain_20261102.json](results/full_retrain_20261102.json), [RESULTS_FULL_RETRAIN.md](RESULTS_FULL_RETRAIN.md). The historical random-CV 0.880/0.989 artifact remains **unsupported as evidence**; the similar AP here is a coincidence of a different design and does not corroborate it.

A pre-authorisation variant (same split/seed/parameters, validation-only cost threshold) excludes every post-transaction balance field. Held-out test: AP **0.8366078463** at a 3.67% fraud rate, AUROC **0.9896117104**, precision 0.8333 / recall 0.9906 at threshold 0.99999998. Full record and reproduction: [results/preauth_20261102.json](results/preauth_20261102.json), [RESULTS_PREAUTH.md](RESULTS_PREAUTH.md). Quote AP 0.837 for pre-authorisation blocking and AP 0.880 only for post-transaction scoring; the historical 0.880/0.989 remains unsupported.
