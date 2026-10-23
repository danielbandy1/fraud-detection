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
    """History from strictly EARLIER steps. Same-hour ordering is unknown.

    AUTHORED BY ORION: replaces full-window counts that leaked future records.
    Returns original row order/index so inference probabilities remain aligned.
    A batch must contain history; isolated records have zero observed history.
    """
    from collections import defaultdict
    df = df.copy()
    n = len(df)
    out = {c: np.zeros(n, dtype=float) for c in
           ['orig_tx_count','orig_unique_dest','orig_step_gap','dest_recv_count']}
    out['orig_step_gap'][:] = 999
    sent, received = defaultdict(int), defaultdict(int)
    destinations, last = defaultdict(set), {}
    steps = pd.to_numeric(df['step'], errors='raise').to_numpy()
    origins, dests = df['nameorig'].to_numpy(), df['namedest'].to_numpy()
    order = np.argsort(steps, kind='stable')
    start = 0
    while start < n:
        end = start + 1
        while end < n and steps[order[end]] == steps[order[start]]: end += 1
        positions = order[start:end]
        for pos in positions:
            origin, dest = origins[pos], dests[pos]
            out['orig_tx_count'][pos] = sent[origin]
            out['orig_unique_dest'][pos] = len(destinations[origin])
            out['dest_recv_count'][pos] = received[dest]
            if origin in last:
                out['orig_step_gap'][pos] = min(float(steps[pos]) - last[origin],999)
        for pos in positions:
            origin, dest = origins[pos], dests[pos]
            sent[origin] += 1; destinations[origin].add(dest); received[dest] += 1
            last[origin] = float(steps[pos])
        start = end
    for col, values in out.items(): df[col] = values
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
