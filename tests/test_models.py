import sys
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models import evaluate, optimal_threshold, EvalMetrics


def make_perfect_probs(n=1000, fraud_rate=0.05):
    rng = np.random.default_rng(0)
    y = (rng.random(n) < fraud_rate).astype(int)
    probs = y.astype(float) + rng.normal(0, 0.05, n)
    probs = np.clip(probs, 0, 1)
    return y, probs


def make_random_probs(n=1000, fraud_rate=0.05):
    rng = np.random.default_rng(1)
    y = (rng.random(n) < fraud_rate).astype(int)
    probs = rng.random(n)
    return y, probs


def test_evaluate_returns_metrics():
    y, probs = make_perfect_probs()
    m = evaluate(y, probs)
    assert isinstance(m, EvalMetrics)
    assert 0 <= m.auprc <= 1
    assert 0 <= m.auroc <= 1
    assert 0 <= m.f1 <= 1


def test_evaluate_good_model_high_auprc():
    y, probs = make_perfect_probs()
    m = evaluate(y, probs)
    assert m.auprc > 0.7, f"Expected AUPRC > 0.7, got {m.auprc:.4f}"


def test_evaluate_random_model_low_auprc():
    y, probs = make_random_probs()
    m = evaluate(y, probs)
    assert m.auprc < 0.3, f"Expected low AUPRC for random model, got {m.auprc:.4f}"


def test_evaluate_confusion_matrix_sums():
    y, probs = make_perfect_probs()
    m = evaluate(y, probs)
    assert m.tp + m.fp + m.fn + m.tn == len(y)


def test_optimal_threshold_returns_float():
    y, probs = make_perfect_probs()
    t = optimal_threshold(y, probs)
    assert isinstance(t, float)
    assert 0 < t < 1


def test_optimal_threshold_high_fn_cost_lowers_threshold():
    y, probs = make_perfect_probs()
    t_strict = optimal_threshold(y, probs, cost_fn=50, cost_fp=1)
    t_lenient = optimal_threshold(y, probs, cost_fn=1, cost_fp=1)
    assert t_strict <= t_lenient, "Higher FN cost should yield lower (more aggressive) threshold"


def test_evaluate_with_explicit_threshold():
    y, probs = make_perfect_probs()
    m = evaluate(y, probs, threshold=0.5)
    assert m.threshold == 0.5


def test_expected_loss_positive():
    y, probs = make_perfect_probs()
    m = evaluate(y, probs)
    assert m.expected_loss >= 0
