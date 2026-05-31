#!/usr/bin/env python3
"""
Optuna hyperparameter tuning for the fraud detection LightGBM model.
Designed to run on MCC SLURM — each trial is independent, results stored in SQLite.
"""
import os
import sys
import json
import argparse
import time
import random
import numpy as np
import pandas as pd
import optuna
import lightgbm as lgb
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).parent))
from src.features import load_raw, clean, build_features, FEATURE_COLS, TARGET_COL

optuna.logging.set_verbosity(optuna.logging.WARNING)


def objective(trial: optuna.Trial, X: pd.DataFrame, y: pd.Series) -> float:
    scale = (y == 0).sum() / (y == 1).sum()

    params = {
        "objective": "binary",
        "metric": "average_precision",
        "scale_pos_weight": scale,
        "verbose": -1,
        "n_jobs": 4,
        "seed": trial.number,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 31, 255),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
    }

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    auprc_scores = []

    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            valid_sets=[dval],
            callbacks=[
                lgb.early_stopping(40, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )
        preds = model.predict(X_val)
        auprc_scores.append(average_precision_score(y_val, preds))

    return float(np.mean(auprc_scores))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--storage", type=str, default="sqlite:///models/optuna.db")
    parser.add_argument("--study-name", type=str, default="fraud-lgbm")
    parser.add_argument("--sample-frac", type=float, default=0.3,
                        help="Fraction of data to use per trial (speeds up tuning)")
    args = parser.parse_args()

    Path("models").mkdir(exist_ok=True)

    print("Loading data...")
    raw = load_raw()
    df = clean(raw)
    df = build_features(df)

    if args.sample_frac < 1.0:
        fraud = df[df[TARGET_COL] == 1].sample(frac=args.sample_frac, random_state=42)
        legit = df[df[TARGET_COL] == 0].sample(frac=args.sample_frac, random_state=42)
        df = pd.concat([fraud, legit]).sample(frac=1.0, random_state=42).reset_index(drop=True)
        print(f"  Sampled to {len(df):,} rows ({args.sample_frac:.0%})")

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    print(f"  Fraud rate: {y.mean():.4%}  |  n={len(y):,}")

    # Stagger startup to avoid SQLite alembic_version race on shared DB
    time.sleep(random.uniform(0, int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")) * 3))
    study = optuna.create_study(
        direction="maximize",
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
    )

    study.optimize(
        lambda trial: objective(trial, X, y),
        n_trials=args.n_trials,
        show_progress_bar=False,
    )

    best = study.best_trial
    print(f"\nBest AUPRC: {best.value:.4f}")
    print(f"Best params: {json.dumps(best.params, indent=2)}")

    with open("models/best_params.json", "w") as f:
        json.dump({"auprc": best.value, "params": best.params}, f, indent=2)
    print("Saved to models/best_params.json")


if __name__ == "__main__":
    main()
