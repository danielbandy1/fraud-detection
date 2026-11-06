import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    average_precision_score, precision_recall_curve,
    roc_auc_score, f1_score, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb
from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class EvalMetrics:
    auprc: float
    auroc: float
    f1: float
    precision: float
    recall: float
    threshold: float
    tn: int
    fp: int
    fn: int
    tp: int
    expected_loss: float  # business cost at optimal threshold


def optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray,
                      cost_fn: float = 1.0, cost_fp: float = 0.1) -> float:
    """Minimise cost_fn*FN + cost_fp*FP on VALIDATION labels only.

    Includes the no-alert operating point (> maximum score). In equal-cost
    ties choose the lowest threshold. AUTHORED BY ORION.
    """
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob, dtype=float)
    if not (y_true.ndim == y_prob.ndim == 1 and len(y_true) == len(y_prob) and len(y_true)):
        raise ValueError("Aligned nonempty outcome/probability vectors required")
    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("Binary outcomes required")
    if not np.isfinite(y_prob).all() or ((y_prob < 0) | (y_prob > 1)).any():
        raise ValueError("Probabilities must be finite and within [0,1]")
    if not np.isfinite([cost_fn, cost_fp]).all() or min(cost_fn, cost_fp) < 0:
        raise ValueError("Costs must be finite and nonnegative")
    order = np.argsort(y_prob, kind="stable")
    scores, labels = y_prob[order], y_true[order]
    thresholds, starts = np.unique(scores, return_index=True)
    positives = np.cumsum(labels[::-1])[::-1][starts]
    negatives = len(labels) - starts - positives
    fn = labels.sum() - positives
    costs = cost_fn * fn + cost_fp * negatives
    thresholds = np.r_[thresholds, np.nextafter(scores[-1], np.inf)]
    costs = np.r_[costs, cost_fn * labels.sum()]
    return float(thresholds[np.argmin(costs)])


def evaluate(y_true: np.ndarray, y_prob: np.ndarray, threshold: float | None = None) -> EvalMetrics:
    if threshold is None:
        threshold = optimal_threshold(y_true, y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    cost_fn, cost_fp = 1.0, 0.1
    expected_loss = cost_fn * fn + cost_fp * fp
    return EvalMetrics(
        auprc=average_precision_score(y_true, y_prob),
        auroc=roc_auc_score(y_true, y_prob),
        f1=f1_score(y_true, y_pred),
        precision=tp / (tp + fp + 1e-9),
        recall=tp / (tp + fn + 1e-9),
        threshold=threshold,
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
        expected_loss=expected_loss,
    )


def train_lgbm(X_train: pd.DataFrame, y_train: pd.Series,
               X_val: pd.DataFrame, y_val: pd.Series,
               params: Dict[str, Any] | None = None) -> lgb.Booster:
    scale = (y_train == 0).sum() / (y_train == 1).sum()
    default_params = {
        "objective": "binary",
        "metric": "average_precision",
        "scale_pos_weight": scale,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": 50,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "n_jobs": -1,
        "seed": 42,
    }
    if params:
        default_params.update(params)

    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
    model = lgb.train(
        default_params,
        dtrain,
        num_boost_round=1000,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)],
    )
    return model


def train_logistic(X_train: pd.DataFrame, y_train: pd.Series) -> tuple:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    model.fit(X_scaled, y_train)
    return model, scaler


def cross_validate_lgbm(X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> dict:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_probs = np.zeros(len(y))
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model = train_lgbm(X_tr, y_tr, X_val, y_val)
        oof_probs[val_idx] = model.predict(X_val)
        fold_metrics.append(evaluate(y_val.values, oof_probs[val_idx]))

    overall = evaluate(y.values, oof_probs)
    return {
        "oof_probs": oof_probs,
        "overall": overall,
        "fold_metrics": fold_metrics,
    }
