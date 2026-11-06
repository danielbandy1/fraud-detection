"""Inference helper for PaySim fraud transactions."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

from src.features import build_features, FEATURE_COLS

REQUIRED_RAW_COLUMNS = [
    "step", "type", "amount", "nameorig", "oldbalanceorg", "newbalanceorig",
    "namedest", "oldbalancedest", "newbalancedest",
]


def _as_dataframe(transactions) -> pd.DataFrame:
    if isinstance(transactions, pd.DataFrame):
        df = transactions.copy()
    elif isinstance(transactions, dict):
        df = pd.DataFrame([transactions])
    else:
        df = pd.DataFrame(transactions)
    df.columns = [str(c).lower() for c in df.columns]
    if df.empty:
        return df
    missing = [c for c in REQUIRED_RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required raw transaction columns: {missing}")
    if "isfraud" not in df.columns:
        df["isfraud"] = 0
    if "isflaggedfraud" not in df.columns:
        df["isflaggedfraud"] = 0
    return df


def load_fraud_model(model_path: str | Path | None = None) -> lgb.Booster:
    """Default to the verified evaluated model; explicit paths are caller-managed."""
    if model_path is None:
        return load_evaluated_fraud_model()[0]
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Fraud model not found: {path}")
    return lgb.Booster(model_file=str(path))


def predict(transactions, model=None) -> np.ndarray:
    """Return fraud probabilities for one or more raw PaySim transactions."""
    df = _as_dataframe(transactions)
    if df.empty:
        return np.array([], dtype=float)
    X = build_features(df)[FEATURE_COLS]
    model = model or load_fraud_model()
    probs = model.predict(X)
    return np.asarray(probs, dtype=float)


def load_evaluated_fraud_model(root: str | Path | None = None):
    """Load the hash-bound forward-time artifact and validation threshold.

    AUTHORED BY ORION. Returns (model, threshold, provenance); no historical fallback.
    """
    import hashlib, json
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    report_path = root / 'FULL_EVAL_ORION_20261024.json'
    report_bytes = report_path.read_bytes()
    report = json.loads(report_bytes)
    model_path = root / 'full_eval_results/model.txt'
    model_bytes = model_path.read_bytes()
    if hashlib.sha256(model_bytes).hexdigest() != report['files']['model.txt']:
        raise ValueError('Evaluated model hash mismatch')
    if hashlib.sha256((root/'src/features.py').read_bytes()).hexdigest() != report['features_sha256']:
        raise ValueError('Evaluated feature implementation hash mismatch')
    if report['features'] != FEATURE_COLS:
        raise ValueError('Evaluated feature order mismatch')
    selection = report['threshold_selection']
    threshold = float(selection['threshold'])
    if selection['test_used_for_selection'] is not False or not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('Invalid validation threshold provenance')
    # Parse exactly the verified bytes rather than reopening a potentially changed file.
    model = lgb.Booster(model_str=model_bytes.decode('utf-8'))
    if model.feature_name() != FEATURE_COLS:
        raise ValueError('Model feature names mismatch')
    provenance = dict(model_sha256=report['files']['model.txt'],
                      evaluation_report_sha256=hashlib.sha256(report_bytes).hexdigest(),
                      scope='Forward-time public PaySim post-transaction benchmark; partial supplied history; not validated deployment performance')
    return model, threshold, provenance
