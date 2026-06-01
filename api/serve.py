#!/usr/bin/env python3
"""
FastAPI inference endpoint — PaySim financial fraud detection.

Run:
    uvicorn api.serve:app --host 0.0.0.0 --port 8001

Design note: graph features (orig_tx_count, orig_step_gap, dest_recv_count,
orig_unique_dest) require historical context. Single-transaction /score calls
use conservative "first-time sender" defaults; /score/window accepts a list of
recent transactions for the account so graph features compute correctly.
"""
from __future__ import annotations
import json
import pathlib
import sys
from contextlib import asynccontextmanager

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import numpy as np
import lightgbm as lgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.features import build_features, FEATURE_COLS
from src.predict import _as_dataframe

MODEL_PATH   = pathlib.Path(__file__).parent.parent / "models" / "lgbm_fraud.txt"
RESULTS_PATH = pathlib.Path(__file__).parent.parent / "models" / "results.json"

_model: lgb.Booster | None = None
_explainer = None
_threshold: float = 0.942  # fallback; overridden from results.json at startup


def _load():
    global _model, _explainer, _threshold
    if _model is None:
        _model = lgb.Booster(model_file=str(MODEL_PATH))
        if RESULTS_PATH.exists():
            with open(RESULTS_PATH) as f:
                _threshold = json.load(f).get("lgbm", {}).get("threshold", _threshold)
        try:
            import shap
            _explainer = shap.TreeExplainer(_model)
        except Exception:
            _explainer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load()
    yield


app = FastAPI(
    title="PaySim Fraud Detection",
    description=(
        "Scores mobile money transactions for fraud risk using LightGBM + graph features. "
        "Use /score/window for accurate scoring — it requires recent account history "
        "so graph features (velocity, fan-out, step gap) compute correctly."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


class Transaction(BaseModel):
    step: int       = Field(..., description="Simulation hour (1 step = 1 hour)")
    type: str       = Field(..., description="CASH_OUT or TRANSFER")
    amount: float   = Field(..., gt=0)
    nameorig: str   = Field(..., description="Originator account ID")
    oldbalanceorg: float   = Field(..., ge=0)
    newbalanceorig: float  = Field(..., ge=0)
    namedest: str   = Field(..., description="Destination account ID")
    oldbalancedest: float  = Field(..., ge=0)
    newbalancedest: float  = Field(..., ge=0)


class FraudScore(BaseModel):
    fraud_probability: float
    risk_tier: str
    is_flagged: bool
    top_risk_factors: list[dict]
    graph_features_available: bool = False
    model_threshold: float


class BatchResponse(BaseModel):
    predictions: list[FraudScore]
    count: int
    flagged_count: int


def _score_df(rows: list[dict], target_idx: int = -1) -> FraudScore:
    """Score transaction at target_idx using all rows for graph feature context."""
    _load()
    for r in rows:
        r.setdefault("isfraud", 0)
        r.setdefault("isflaggedfraud", 0)

    try:
        df = _as_dataframe(rows)
        df_feat = build_features(df)
        X_all = df_feat[FEATURE_COLS]
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Feature error: {e}")

    X = X_all.iloc[[target_idx]]
    prob = float(_model.predict(X)[0])
    flagged = prob >= _threshold
    tier = "HIGH" if prob >= 0.80 else ("MEDIUM" if prob >= _threshold else "LOW")
    has_graph = len(rows) > 1

    factors = []
    if _explainer is not None:
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sv = _explainer.shap_values(X)
            arr = np.array(sv[1] if isinstance(sv, list) else sv)[0]
            factors = sorted(
                [{"feature": f, "shap": round(float(s), 4)} for f, s in zip(FEATURE_COLS, arr)],
                key=lambda x: abs(x["shap"]),
                reverse=True,
            )[:5]
        except Exception:
            pass

    return FraudScore(
        fraud_probability=round(prob, 4),
        risk_tier=tier,
        is_flagged=flagged,
        top_risk_factors=factors,
        graph_features_available=has_graph,
        model_threshold=_threshold,
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "model_path": str(MODEL_PATH),
        "fraud_threshold": _threshold,
    }


@app.post("/score", response_model=FraudScore)
def score(tx: Transaction):
    """
    Score a single transaction. Graph features default to 'first-time sender' values.
    For accurate scoring, use /score/window with recent account history.
    """
    return _score_df([tx.model_dump()])


@app.post("/score/window", response_model=FraudScore)
def score_window(transactions: list[Transaction]):
    """
    Score the LAST transaction in the list using all transactions for graph context.
    Pass recent history for the same originator account first, target transaction last.
    Minimum 1 transaction; maximum 500.
    """
    if not transactions:
        raise HTTPException(status_code=400, detail="At least one transaction required.")
    if len(transactions) > 500:
        raise HTTPException(status_code=400, detail="Window limit is 500 transactions.")
    rows = [tx.model_dump() for tx in transactions]
    return _score_df(rows, target_idx=-1)


@app.post("/score/batch", response_model=BatchResponse)
def score_batch(transactions: list[Transaction]):
    """Score each transaction independently (no cross-transaction graph context)."""
    if len(transactions) > 1000:
        raise HTTPException(status_code=400, detail="Batch limit is 1000 transactions.")
    results = [_score_df([tx.model_dump()]) for tx in transactions]
    flagged = sum(1 for r in results if r.is_flagged)
    return BatchResponse(predictions=results, count=len(results), flagged_count=flagged)
