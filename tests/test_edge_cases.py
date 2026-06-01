import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from src.features import FEATURE_COLS
from src.predict import predict


class DummyModel:
    def predict(self, X):
        assert list(X.columns) == FEATURE_COLS
        return np.linspace(0.0, 0.9, len(X))


def base_transaction(**overrides):
    row = {
        "step": 1,
        "type": "CASH_OUT",
        "amount": 100.0,
        "nameorig": "C001",
        "oldbalanceorg": 1000.0,
        "newbalanceorig": 900.0,
        "namedest": "M001",
        "oldbalancedest": 0.0,
        "newbalancedest": 100.0,
        "isfraud": 0,
        "isflaggedfraud": 0,
    }
    row.update(overrides)
    return row


def test_predict_all_zero_transaction():
    row = base_transaction(amount=0.0, oldbalanceorg=0.0, newbalanceorig=0.0, oldbalancedest=0.0, newbalancedest=0.0)
    probs = predict([row], model=DummyModel())
    assert probs.shape == (1,)
    assert np.isfinite(probs).all()


def test_predict_single_transaction():
    probs = predict(base_transaction(), model=DummyModel())
    assert probs.tolist() == [0.0]


def test_predict_duplicate_transaction_ids_or_rows():
    row = base_transaction(nameorig="C_DUP", namedest="M_DUP")
    probs = predict([row, row.copy()], model=DummyModel())
    assert len(probs) == 2


def test_predict_missing_features_raises_clear_error():
    row = base_transaction()
    row.pop("amount")
    with pytest.raises(ValueError, match="Missing required raw transaction columns"):
        predict([row], model=DummyModel())
