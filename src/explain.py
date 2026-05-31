import shap
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")


def get_shap_values(model: lgb.Booster, X: pd.DataFrame) -> shap.Explanation:
    explainer = shap.TreeExplainer(model)
    return explainer(X)


def summary_plot(shap_values: shap.Explanation, X: pd.DataFrame, max_display: int = 15) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 6))
    shap.summary_plot(shap_values, X, max_display=max_display, show=False, plot_size=None)
    fig = plt.gcf()
    fig.tight_layout()
    return fig


def waterfall_plot(shap_values: shap.Explanation, idx: int) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 5))
    shap.waterfall_plot(shap_values[idx], show=False)
    fig = plt.gcf()
    fig.tight_layout()
    return fig


def feature_importance_df(model: lgb.Booster, feature_names: list) -> pd.DataFrame:
    gain = model.feature_importance(importance_type="gain")
    split = model.feature_importance(importance_type="split")
    return (
        pd.DataFrame({"feature": feature_names, "gain": gain, "split": split})
        .sort_values("gain", ascending=False)
        .reset_index(drop=True)
    )
