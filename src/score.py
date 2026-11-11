#!/usr/bin/env python3
"""AUTHORED BY GEMINI · November 2026.

Standalone production real-time inference scorer for PaySim financial fraud detection.
Provides sub-50ms deterministic scoring, risk tiering, action recommendations,
and SHAP local explainability without requiring a web server daemon.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time
from typing import Any

import numpy as np
import pandas as pd

# Ensure repo root is on sys.path
ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features import FEATURE_COLS, build_features
from src.predict import _as_dataframe, load_evaluated_fraud_model

# Golden benchmark transactions from the verified held-out test window (step > 600)
BENCHMARK_LEGIT_TX: dict[str, Any] = {
    "step": 601,
    "type": "CASH_OUT",
    "amount": 264164.14,
    "nameorig": "C759019714",
    "oldbalanceorg": 7461.45,
    "newbalanceorig": 0.0,
    "namedest": "C1934185092",
    "oldbalancedest": 4813833.71,
    "newbalancedest": 5077997.86,
}

BENCHMARK_FRAUD_TX: dict[str, Any] = {
    "step": 601,
    "type": "TRANSFER",
    "amount": 414973.48,
    "nameorig": "C1903805843",
    "oldbalanceorg": 414973.48,
    "newbalanceorig": 0.0,
    "namedest": "C2028972302",
    "oldbalancedest": 0.0,
    "newbalancedest": 0.0,
}


class FinancialFraudScorer:
    """Production inference scorer for PaySim mobile money fraud detection."""

    def __init__(self, root: pathlib.Path | str | None = None):
        self.root = pathlib.Path(root) if root is not None else ROOT
        self.model, self.threshold, self.provenance = load_evaluated_fraud_model(self.root)
        self.feature_cols = list(FEATURE_COLS)

        # Initialize TreeExplainer for local feature attributions
        try:
            import shap
            self.explainer = shap.TreeExplainer(self.model)
        except Exception:
            self.explainer = None

    def _prepare_features(
        self,
        transaction: dict[str, Any],
        history: list[dict[str, Any]] | None = None,
    ) -> tuple[pd.DataFrame, bool]:
        """Transform raw transaction(s) into model feature vector with strictly earlier graph context."""
        if history:
            rows = [dict(h) for h in history] + [dict(transaction)]
        else:
            rows = [dict(transaction)]

        df = _as_dataframe(rows)
        df_feat = build_features(df)
        X = df_feat[self.feature_cols]

        # Determine whether earlier history contributed graph context to target
        target_feat = df_feat.iloc[-1]
        has_graph = bool(
            target_feat["orig_tx_count"] > 0 or target_feat["dest_recv_count"] > 0
        )

        return X.iloc[[-1]], has_graph

    def predict(
        self,
        transaction: dict[str, Any],
        history: list[dict[str, Any]] | None = None,
        explain: bool = True,
    ) -> dict[str, Any]:
        """Score a single transaction with sub-50ms latency and operational triage.

        Parameters
        ----------
        transaction : dict
            Raw transaction payload with step, type, amount, nameorig, oldbalanceorg, etc.
        history : list of dict, optional
            Preceding transactions in the audit window for rolling graph features.
        explain : bool
            Whether to compute local SHAP feature attributions.

        Returns
        -------
        dict containing fraud_probability, is_flagged, risk_tier, action, top_risk_factors, latency_ms.
        """
        t0 = time.perf_counter()

        X, has_graph = self._prepare_features(transaction, history=history)
        prob = float(self.model.predict(X, num_threads=2)[0])
        flagged = bool(prob >= self.threshold)

        # Operational Risk Tiering & Decision Rules
        if prob >= 0.99:
            tier = "CRITICAL"
            action = "BLOCK_TRANSACTION"
        elif prob >= self.threshold:
            tier = "HIGH"
            action = "BLOCK_TRANSACTION"
        elif prob >= 0.50:
            tier = "MEDIUM"
            action = "STEP_UP_AUTH"
        else:
            tier = "LOW"
            action = "AUTO_APPROVE"

        # Local explainability via TreeExplainer SHAP
        top_factors = []
        if explain and self.explainer is not None:
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    sv = self.explainer.shap_values(X)
                arr = np.array(sv[1] if isinstance(sv, list) else sv)[0]
                top_factors = sorted(
                    [
                        {"feature": f, "shap": round(float(s), 4)}
                        for f, s in zip(self.feature_cols, arr)
                    ],
                    key=lambda item: abs(item["shap"]),
                    reverse=True,
                )[:5]
            except Exception:
                top_factors = []

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        return {
            "fraud_probability": round(prob, 6),
            "is_flagged": flagged,
            "risk_tier": tier,
            "action": action,
            "threshold": round(self.threshold, 6),
            "graph_features_available": has_graph,
            "top_risk_factors": top_factors,
            "latency_ms": latency_ms,
        }

    def predict_batch(
        self,
        transactions: list[dict[str, Any]],
        explain: bool = False,
    ) -> list[dict[str, Any]]:
        """Score a list of transactions independently."""
        return [self.predict(t, explain=explain) for t in transactions]

    def benchmark_latency(self, n_iterations: int = 25) -> dict[str, float]:
        """Benchmark fast-path and explain-mode latency distributions in milliseconds."""
        # Warmup
        self.predict(BENCHMARK_LEGIT_TX, explain=False)

        fast_latencies = []
        for _ in range(n_iterations):
            t0 = time.perf_counter()
            self.predict(BENCHMARK_LEGIT_TX, explain=False)
            fast_latencies.append((time.perf_counter() - t0) * 1000.0)

        explain_latencies = []
        for _ in range(n_iterations):
            t0 = time.perf_counter()
            self.predict(BENCHMARK_LEGIT_TX, explain=True)
            explain_latencies.append((time.perf_counter() - t0) * 1000.0)

        return {
            "fast_p50_ms": round(float(np.percentile(fast_latencies, 50)), 2),
            "fast_p95_ms": round(float(np.percentile(fast_latencies, 95)), 2),
            "explain_p50_ms": round(float(np.percentile(explain_latencies, 50)), 2),
            "explain_p95_ms": round(float(np.percentile(explain_latencies, 95)), 2),
        }


def main():
    parser = argparse.ArgumentParser(description="PaySim Standalone Real-Time Scorer")
    parser.add_argument("--demo", action="store_true", help="Run demonstration on benchmark transactions")
    parser.add_argument("--benchmark", action="store_true", help="Run latency benchmark suite")
    args = parser.parse_args()

    scorer = FinancialFraudScorer()
    print("=" * 70)
    print("PaySim Financial Fraud Standalone Real-Time Scorer")
    print(f"Model: LightGBM (18 trees) | Features: {len(scorer.feature_cols)}")
    print(f"Validation-tuned F1 Threshold: {scorer.threshold:.6f}")
    print(f"Model SHA256: {scorer.provenance['model_sha256'][:16]}...")
    print("=" * 70)

    if args.demo or not (args.demo or args.benchmark):
        print("\n--- Scoring Benchmark Legitimate Transaction ---")
        res_legit = scorer.predict(BENCHMARK_LEGIT_TX, explain=True)
        print(f"Amount: ${BENCHMARK_LEGIT_TX['amount']:,.2f} ({BENCHMARK_LEGIT_TX['type']})")
        print(f"Fraud Probability: {res_legit['fraud_probability']:.6f}")
        print(f"Flagged: {res_legit['is_flagged']} | Tier: {res_legit['risk_tier']} | Action: {res_legit['action']}")
        print(f"Latency: {res_legit['latency_ms']} ms")
        print("Top 5 Driving Risk Factors (SHAP):")
        for f in res_legit["top_risk_factors"]:
            print(f"  - {f['feature']:<25}: {f['shap']:+.4f}")

        print("\n--- Scoring Benchmark Fraudulent Transaction ---")
        res_fraud = scorer.predict(BENCHMARK_FRAUD_TX, explain=True)
        print(f"Amount: ${BENCHMARK_FRAUD_TX['amount']:,.2f} ({BENCHMARK_FRAUD_TX['type']})")
        print(f"Fraud Probability: {res_fraud['fraud_probability']:.6f}")
        print(f"Flagged: {res_fraud['is_flagged']} | Tier: {res_fraud['risk_tier']} | Action: {res_fraud['action']}")
        print(f"Latency: {res_fraud['latency_ms']} ms")
        print("Top 5 Driving Risk Factors (SHAP):")
        for f in res_fraud["top_risk_factors"]:
            print(f"  - {f['feature']:<25}: {f['shap']:+.4f}")

        assert res_fraud["fraud_probability"] > res_legit["fraud_probability"], "Fraud must score higher than legit"
        assert res_fraud["is_flagged"] is True, "Fraud must be flagged"
        assert res_legit["is_flagged"] is False, "Legit must not be flagged"
        print("\n[PASS] Verification: Fraud scored strictly higher than legitimate transaction and triggered BLOCK action.")

    if args.benchmark:
        print("\n--- Running Latency Benchmarks (25 iterations) ---")
        bench = scorer.benchmark_latency(n_iterations=25)
        for k, v in bench.items():
            print(f"  {k:<20}: {v:.2f} ms")
        assert bench["fast_p95_ms"] < 50.0, f"Fast P95 latency {bench['fast_p95_ms']} ms exceeded 50ms SLA"
        print("[PASS] Sub-50ms latency SLA confirmed.")


if __name__ == "__main__":
    main()
