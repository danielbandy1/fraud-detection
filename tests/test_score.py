#!/usr/bin/env python3
"""AUTHORED BY GEMINI · November 2026.

Unit tests for standalone PaySim financial fraud real-time inference scorer.
Tests:
1. Model loading, feature schema, and hash-bound provenance.
2. Prediction bounds (0 <= prob <= 1) and valid operational risk tiers/actions.
3. Deterministic output for identical transaction inputs.
4. Relative risk separation (real test fraud transaction vs legitimate transaction).
5. Audit window history integration for rolling graph features.
6. Batch scoring contract and consistency.
7. Local SHAP factor attributions.
8. Sub-50ms inference latency benchmark SLA.
"""
from __future__ import annotations

import pathlib
import sys
import time

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features import FEATURE_COLS
from src.score import (
    FinancialFraudScorer,
    BENCHMARK_LEGIT_TX,
    BENCHMARK_FRAUD_TX,
)


@pytest.fixture(scope="module")
def scorer():
    return FinancialFraudScorer()


def test_scorer_initialization(scorer):
    assert scorer.model is not None
    assert len(scorer.feature_cols) == 19
    assert scorer.feature_cols == FEATURE_COLS
    assert 0.0 < scorer.threshold < 1.0
    assert "model_sha256" in scorer.provenance
    assert "evaluation_report_sha256" in scorer.provenance


def test_predict_schema_and_bounds(scorer):
    res = scorer.predict(BENCHMARK_LEGIT_TX, explain=True)
    assert 0.0 <= res["fraud_probability"] <= 1.0
    assert isinstance(res["is_flagged"], bool)
    assert res["risk_tier"] in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
    assert res["action"] in {"AUTO_APPROVE", "STEP_UP_AUTH", "BLOCK_TRANSACTION"}
    assert res["threshold"] == round(scorer.threshold, 6)
    assert isinstance(res["latency_ms"], float)
    assert len(res["top_risk_factors"]) == 5


def test_predict_determinism(scorer):
    res1 = scorer.predict(BENCHMARK_LEGIT_TX, explain=True)
    res2 = scorer.predict(BENCHMARK_LEGIT_TX, explain=True)
    assert res1["fraud_probability"] == res2["fraud_probability"]
    assert res1["is_flagged"] == res2["is_flagged"]
    assert res1["risk_tier"] == res2["risk_tier"]
    assert res1["action"] == res2["action"]
    assert res1["top_risk_factors"] == res2["top_risk_factors"]


def test_benchmark_risk_separation(scorer):
    res_legit = scorer.predict(BENCHMARK_LEGIT_TX, explain=False)
    res_fraud = scorer.predict(BENCHMARK_FRAUD_TX, explain=False)

    # Fraud probability must decisively exceed legit probability
    assert res_fraud["fraud_probability"] > res_legit["fraud_probability"]
    assert res_fraud["fraud_probability"] >= 0.99
    assert res_legit["fraud_probability"] < 0.10

    # Operational triage assertions
    assert res_fraud["is_flagged"] is True
    assert res_fraud["action"] == "BLOCK_TRANSACTION"
    assert res_fraud["risk_tier"] in {"HIGH", "CRITICAL"}

    assert res_legit["is_flagged"] is False
    assert res_legit["action"] == "AUTO_APPROVE"
    assert res_legit["risk_tier"] == "LOW"


def test_history_window_graph_features(scorer):
    # Standalone transaction without history -> graph_features_available should be False
    res_isolated = scorer.predict(BENCHMARK_LEGIT_TX, history=None)
    assert res_isolated["graph_features_available"] is False

    # Transaction with earlier-step history for same originator
    history_tx = {
        "step": BENCHMARK_LEGIT_TX["step"] - 10,
        "type": "CASH_OUT",
        "amount": 5000.0,
        "nameorig": BENCHMARK_LEGIT_TX["nameorig"],
        "oldbalanceorg": 50000.0,
        "newbalanceorig": 45000.0,
        "namedest": "M9999",
        "oldbalancedest": 0.0,
        "newbalancedest": 5000.0,
    }
    res_windowed = scorer.predict(BENCHMARK_LEGIT_TX, history=[history_tx])
    assert res_windowed["graph_features_available"] is True


def test_batch_prediction(scorer):
    batch = [BENCHMARK_LEGIT_TX, BENCHMARK_FRAUD_TX]
    results = scorer.predict_batch(batch, explain=False)
    assert len(results) == 2
    assert results[0]["is_flagged"] is False
    assert results[1]["is_flagged"] is True


def test_shap_factors_content(scorer):
    res = scorer.predict(BENCHMARK_FRAUD_TX, explain=True)
    factors = res["top_risk_factors"]
    assert len(factors) == 5
    for item in factors:
        assert "feature" in item
        assert item["feature"] in scorer.feature_cols
        assert "shap" in item
        assert isinstance(item["shap"], float)


def test_sub_50ms_latency_sla(scorer):
    # Warmup
    scorer.predict(BENCHMARK_LEGIT_TX, explain=False)

    latencies = []
    for _ in range(25):
        t0 = time.perf_counter()
        scorer.predict(BENCHMARK_LEGIT_TX, explain=False)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
    print(f"\nP95 Fast-path Latency: {p95_latency:.2f} ms")
    assert p95_latency < 50.0, f"P95 latency {p95_latency:.2f} ms exceeded 50ms SLA"

    # Explain mode latency on ARM SBC
    t_exp0 = time.perf_counter()
    scorer.predict(BENCHMARK_LEGIT_TX, explain=True)
    exp_ms = (time.perf_counter() - t_exp0) * 1000.0
    print(f"Explain-mode Latency: {exp_ms:.2f} ms")
    assert exp_ms < 100.0, f"Explain-mode latency {exp_ms:.2f} ms exceeded 100ms SLA"
