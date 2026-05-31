#!/usr/bin/env python3
"""
Train LightGBM fraud detector on PaySim. Saves model + OOF results to models/.
"""
import sys
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.features import load_raw, clean, build_features, FEATURE_COLS, TARGET_COL
from src.models import train_lgbm, cross_validate_lgbm, train_logistic, evaluate

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


def main():
    print("Loading data...")
    raw = load_raw()
    print(f"  Raw rows: {len(raw):,}")

    df = clean(raw)
    print(f"  After cleaning (CASH_OUT + TRANSFER only): {len(df):,}")
    print(f"  Fraud rate: {df[TARGET_COL].mean():.4%}")

    df = build_features(df)

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    print("\nRunning 5-fold cross-validation (LightGBM)...")
    cv_result = cross_validate_lgbm(X, y, n_splits=5)
    overall = cv_result["overall"]

    print(f"\n--- OOF Results ---")
    print(f"  AUPRC:     {overall.auprc:.4f}")
    print(f"  AUROC:     {overall.auroc:.4f}")
    print(f"  F1:        {overall.f1:.4f}")
    print(f"  Precision: {overall.precision:.4f}")
    print(f"  Recall:    {overall.recall:.4f}")
    print(f"  Threshold: {overall.threshold:.4f}")
    print(f"  TP/FP/FN/TN: {overall.tp}/{overall.fp}/{overall.fn}/{overall.tn}")

    print("\nTraining final model on full data...")
    val_size = int(0.1 * len(X))
    X_tr, X_val = X.iloc[:-val_size], X.iloc[-val_size:]
    y_tr, y_val = y.iloc[:-val_size], y.iloc[-val_size:]
    final_model = train_lgbm(X_tr, y_tr, X_val, y_val)

    print("Training logistic regression baseline...")
    logreg, scaler = train_logistic(X_tr, y_tr)
    lr_probs = logreg.predict_proba(scaler.transform(X_val))[:, 1]
    lr_metrics = evaluate(y_val.values, lr_probs)
    print(f"  Logistic AUPRC: {lr_metrics.auprc:.4f}  AUROC: {lr_metrics.auroc:.4f}")

    # Save artifacts
    final_model.save_model(str(MODELS_DIR / "lgbm_fraud.txt"))
    with open(MODELS_DIR / "logreg.pkl", "wb") as f:
        pickle.dump((logreg, scaler), f)

    oof_df = pd.DataFrame({
        "index": df.index,
        "y_true": y.values,
        "y_prob": cv_result["oof_probs"],
    })
    oof_df.to_parquet(MODELS_DIR / "oof_probs.parquet", index=False)

    results = {
        "lgbm": {
            "auprc": overall.auprc,
            "auroc": overall.auroc,
            "f1": overall.f1,
            "precision": overall.precision,
            "recall": overall.recall,
            "threshold": overall.threshold,
            "tp": overall.tp, "fp": overall.fp,
            "fn": overall.fn, "tn": overall.tn,
        },
        "logistic": {
            "auprc": lr_metrics.auprc,
            "auroc": lr_metrics.auroc,
            "f1": lr_metrics.f1,
        },
        "data": {
            "n_rows": len(df),
            "fraud_rate": float(y.mean()),
            "n_fraud": int(y.sum()),
        },
    }
    with open(MODELS_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nArtifacts saved to {MODELS_DIR}/")
    print("Done.")


if __name__ == "__main__":
    main()
