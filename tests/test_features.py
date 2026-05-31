import sys
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.features import (
    clean, add_balance_features, add_time_features,
    add_graph_features, add_type_encoding, build_features, FEATURE_COLS
)


def make_df(n=100, fraud_rate=0.1, tx_type="CASH_OUT"):
    rng = np.random.default_rng(42)
    n_fraud = int(n * fraud_rate)
    df = pd.DataFrame({
        "step": rng.integers(1, 744, n),
        "type": [tx_type] * n,
        "amount": rng.uniform(100, 50000, n),
        "nameorig": [f"C{i % 20}" for i in range(n)],
        "oldbalanceorg": rng.uniform(0, 100000, n),
        "newbalanceorig": rng.uniform(0, 100000, n),
        "namedest": [f"M{i % 30}" for i in range(n)],
        "oldbalancedest": rng.uniform(0, 50000, n),
        "newbalancedest": rng.uniform(0, 50000, n),
        "isfraud": [1] * n_fraud + [0] * (n - n_fraud),
        "isflaggedfraud": 0,
    })
    return df


def test_clean_keeps_cash_out_and_transfer():
    df = pd.concat([make_df(50, tx_type="CASH_OUT"), make_df(50, tx_type="PAYMENT")])
    result = clean(df)
    assert set(result["type"].unique()) == {"CASH_OUT"}
    assert len(result) == 50


def test_balance_features_columns():
    df = make_df()
    result = add_balance_features(df)
    for col in ["orig_balance_delta", "dest_balance_delta", "orig_drained",
                "dest_no_change", "amount_to_orig_balance", "log_amount"]:
        assert col in result.columns


def test_orig_drained_flag():
    df = make_df(10)
    df.loc[0, "oldbalanceorg"] = 500.0
    df.loc[0, "newbalanceorig"] = 0.0
    result = add_balance_features(df)
    assert result.loc[0, "orig_drained"] == 1


def test_time_features_hour_range():
    df = make_df()
    result = add_time_features(df)
    assert result["hour"].between(0, 23).all()


def test_graph_features_no_nulls():
    df = make_df(200)
    result = add_graph_features(df)
    for col in ["orig_tx_count", "orig_unique_dest", "orig_step_gap", "dest_recv_count"]:
        assert result[col].isna().sum() == 0, f"{col} has nulls"


def test_all_feature_cols_present():
    df = make_df(200)
    df = build_features(df)
    for col in FEATURE_COLS:
        assert col in df.columns, f"Missing feature: {col}"


def test_log_amount_nonnegative():
    df = make_df()
    result = add_balance_features(df)
    assert (result["log_amount"] >= 0).all()


def test_is_night_binary():
    df = make_df()
    result = add_time_features(df)
    assert set(result["is_night"].unique()).issubset({0, 1})
