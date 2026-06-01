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


def load_fraud_model(model_path: str | Path = "models/lgbm_fraud.txt") -> lgb.Booster:
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
