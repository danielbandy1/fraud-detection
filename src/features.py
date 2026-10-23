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
    """Vectorized strict-prior-step histories, preserving input order/index.

    AUTHORED BY ORION. Same semantics as the prior repaired implementation:
    no future rows, no within-step ordering, no target labels. Vectorization
    avoids millions of per-account Python sets on the full PaySim source.
    """
    df = df.copy()
    n = len(df)
    columns = ['orig_tx_count','orig_unique_dest','orig_step_gap','dest_recv_count']
    if not n:
        for col in columns: df[col] = np.array([], dtype=float)
        return df
    order = np.argsort(pd.to_numeric(df['step'],errors='raise').to_numpy(),kind='stable')
    work = df.iloc[order][['nameorig','namedest','step']].reset_index(drop=True)
    origin, dest, step = work['nameorig'], work['namedest'], work['step']
    def restore(col, values):
        array = np.empty(n,dtype=float)
        array[order] = np.asarray(values)
        df[col] = array
    counts = work.groupby('nameorig',sort=False).cumcount()
    restore('orig_tx_count',counts.groupby([origin,step],sort=False).transform('min'))
    first_pair = ~work.duplicated(['nameorig','namedest'])
    cumulative = first_pair.groupby(origin,sort=False).cumsum()
    prior_unique = (cumulative.groupby([origin,step],sort=False).transform('max')
                    -first_pair.groupby([origin,step],sort=False).transform('sum'))
    restore('orig_unique_dest',prior_unique)
    gap = (step-step.groupby(origin,sort=False).shift(1)).fillna(999).clip(upper=999)
    restore('orig_step_gap',gap.groupby([origin,step],sort=False).transform('max'))
    counts = work.groupby('namedest',sort=False).cumcount()
    restore('dest_recv_count',counts.groupby([dest,step],sort=False).transform('min'))
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
