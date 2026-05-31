import pandas as pd
import numpy as np

RAW_FILE = "data/raw/PS_20174392719_1491204439457_log.csv"

TRANSACTION_TYPES = ["CASH_OUT", "PAYMENT", "CASH_IN", "TRANSFER", "DEBIT"]


def load_raw(path: str = RAW_FILE) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    # Fraud only occurs in CASH_OUT and TRANSFER in PaySim
    df = df[df["type"].isin(["CASH_OUT", "TRANSFER"])].copy()
    df = df.drop_duplicates()
    return df.reset_index(drop=True)


def add_balance_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # PaySim column typo: oldbalanceOrg → oldbalanceorg (not oldbalanceorig)
    df["orig_balance_delta"] = df["newbalanceorig"] - df["oldbalanceorg"]
    df["dest_balance_delta"] = df["newbalancedest"] - df["oldbalancedest"]

    df["orig_drained"] = ((df["oldbalanceorg"] > 0) & (df["newbalanceorig"] == 0)).astype(int)
    df["dest_no_change"] = (df["dest_balance_delta"] == 0).astype(int)

    df["amount_to_orig_balance"] = df["amount"] / (df["oldbalanceorg"] + 1)
    df["log_amount"] = np.log1p(df["amount"])
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = df["step"] % 24
    df["day"] = df["step"] // 24
    df["is_night"] = ((df["hour"] >= 0) & (df["hour"] < 6)).astype(int)
    return df


def add_graph_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Velocity and degree features per originator account within the transaction window.
    Real fraud rings show high out-degree and rapid consecutive transactions.
    """
    df = df.copy().sort_values(["nameorig", "step"])

    # Transaction count per sender (out-degree proxy)
    orig_counts = df.groupby("nameorig")["step"].transform("count")
    df["orig_tx_count"] = orig_counts

    # Unique destinations per sender (fan-out — rings send to many mules)
    orig_unique_dest = df.groupby("nameorig")["namedest"].transform("nunique")
    df["orig_unique_dest"] = orig_unique_dest

    # Time since last transaction by same sender (low gap = velocity burst)
    df["orig_prev_step"] = df.groupby("nameorig")["step"].shift(1)
    df["orig_step_gap"] = (df["step"] - df["orig_prev_step"]).fillna(999).clip(upper=999)

    # How many times has this destination been seen (mule re-use)
    dest_recv_count = df.groupby("namedest")["step"].transform("count")
    df["dest_recv_count"] = dest_recv_count

    df = df.drop(columns=["orig_prev_step"])
    return df


def add_type_encoding(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_transfer"] = (df["type"] == "TRANSFER").astype(int)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_balance_features(df)
    df = add_time_features(df)
    df = add_graph_features(df)
    df = add_type_encoding(df)
    return df


FEATURE_COLS = [
    "amount", "log_amount",
    "oldbalanceorg", "newbalanceorig",
    "oldbalancedest", "newbalancedest",
    "orig_balance_delta", "dest_balance_delta",
    "orig_drained", "dest_no_change",
    "amount_to_orig_balance",
    "hour", "day", "is_night",
    "orig_tx_count", "orig_unique_dest", "orig_step_gap",
    "dest_recv_count",
    "is_transfer",
]

TARGET_COL = "isfraud"
