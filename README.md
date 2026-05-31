# Financial Fraud Detection

[![Tests](https://img.shields.io/badge/tests-16%20passed-brightgreen)](#testing)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/dashboard-Streamlit-ff4b4b)](https://streamlit.io)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](#license)

End-to-end fraud detection pipeline on the PaySim synthetic mobile money dataset — covering graph-based feature engineering, LightGBM with class-imbalance handling, business cost threshold calibration, and SHAP explainability. Wrapped in an interactive Streamlit dashboard.

**Dataset:** [PaySim — Synthetic Financial Fraud](https://www.kaggle.com/datasets/ealaxi/paysim1) — 6.36M transactions, 30 simulated days, binary fraud label.

---

## Quick Results

After filtering to the transaction types where fraud actually occurs (`CASH_OUT` and `TRANSFER`), the model achieves AUPRC 0.880 and AUROC 0.989 on out-of-fold predictions — catching 97.6% of fraud at a cost-optimised threshold. The logistic regression baseline reaches AUPRC 0.788, confirming the graph features and non-linear model add meaningful lift.

---

## Key Results

| Metric | Value |
|---|---|
| Raw transactions | 6,362,620 |
| Filtered (CASH_OUT + TRANSFER) | 2,770,409 |
| Fraud rate | 0.30% |
| AUPRC (OOF, LightGBM) | 0.8800 |
| AUROC (OOF, LightGBM) | 0.9894 |
| F1 at optimal threshold | 0.7653 |
| Recall (fraud caught) | 97.6% |
| AUPRC baseline (logistic) | 0.7880 |

**Verdict: Ship with review queue.** The model catches 97.6% of fraud, and the interactive threshold tuner lets a risk team dial precision vs. recall against their actual cost structure rather than a fixed 0.5 cutoff.

---

## Methods

| Component | What it does |
|---|---|
| Balance features | Detect accounts drained to zero, pass-through destinations, amount-to-balance ratio |
| Graph features | Sender fan-out, velocity (step gap), destination re-use — mule network indicators |
| LightGBM | Gradient boosting with `scale_pos_weight` for class imbalance; 5-fold stratified CV |
| Logistic baseline | Scaled logistic regression for AUPRC comparison |
| Cost-optimal threshold | Minimise `cost_fn × FN + cost_fp × FP`; adjustable via dashboard sliders |
| SHAP | TreeExplainer on stratified 2K sample; beeswarm + per-transaction waterfall |

---

## Dashboard

Five tabs covering the full modelling workflow:

**Tabs:** Overview · Model Performance · Threshold Tuning · SHAP Explainability · Data

### Running locally

```bash
git clone https://github.com/danielbandy1/fraud-detection
cd fraud-detection
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python train.py           # ~3 min, saves model to models/
./run_dashboard.sh        # opens at http://localhost:8503
```

---

## Project Structure

```
fraud-detection/
├── app.py                  # Streamlit dashboard
├── run_dashboard.sh        # Launch script
├── train.py                # Training script — CV + final model + baseline
├── requirements.txt
├── src/
│   ├── features.py         # Feature engineering (balance, time, graph)
│   ├── models.py           # LightGBM, logistic, evaluation, threshold optimisation
│   └── explain.py          # SHAP utilities
├── models/                 # Saved artifacts (gitignored)
│   ├── lgbm_fraud.txt
│   ├── logreg.pkl
│   ├── oof_probs.parquet
│   └── results.json
└── tests/
    ├── test_features.py    # 8 unit tests
    └── test_models.py      # 8 unit tests
```

---

## Method Notes

### Why AUPRC over AUROC

At 0.30% fraud rate, AUROC is inflated by the overwhelming negative class — a model that flags nothing still scores 0.997. AUPRC focuses on the precision-recall trade-off in the region that matters: the top of the score distribution where alerts are actually reviewed.

### Graph Features

Real fraud rings operate through money mule networks. A compromised account drains its balance and fans out to multiple mule accounts in rapid succession. The graph features capture this:

- **`orig_unique_dest`** — unique destinations per sender (fan-out indicator)
- **`orig_step_gap`** — time since previous transaction (velocity burst)
- **`dest_recv_count`** — how many senders have sent to this destination (mule re-use)
- **`orig_drained`** — sender account emptied to zero in one shot

These are computed from the transaction log itself with no external graph infrastructure required.

### Threshold Calibration

Standard classifiers output a score, not a decision. The optimal decision boundary depends on the cost structure of the specific business:

$$t^* = \arg\min_t \; c_{FN} \cdot FN(t) + c_{FP} \cdot FP(t)$$

The dashboard exposes this directly — drag the cost sliders and the optimal threshold, confusion matrix, and catch rate all update live.

### Class Imbalance

Handled via `scale_pos_weight = n_negative / n_positive ≈ 337` in LightGBM. No synthetic oversampling (SMOTE) — oversampling can leak temporal structure in time-ordered transaction data.

---

## Testing

```bash
source .venv/bin/activate
pytest tests/ -v
# 16 passed
```

---

## License

MIT
