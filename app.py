import sys
import json
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import lightgbm as lgb
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import precision_recall_curve, roc_curve

sys.path.insert(0, str(Path(__file__).parent))
from src.features import load_raw, clean, build_features, FEATURE_COLS, TARGET_COL
from src.models import evaluate, optimal_threshold
from src.explain import get_shap_values, feature_importance_df

st.set_page_config(page_title="Fraud Detection", page_icon="🔍", layout="wide")
st.title("🔍 Financial Fraud Detection")
st.caption("PaySim synthetic mobile money dataset · LightGBM + graph features + SHAP")

MODELS_DIR = Path("models")

@st.cache_resource
def load_model():
    model = lgb.Booster(model_file=str(MODELS_DIR / "lgbm_fraud.txt"))
    with open(MODELS_DIR / "logreg.pkl", "rb") as f:
        logreg, scaler = pickle.load(f)
    return model, logreg, scaler

@st.cache_data
def load_data():
    from src.features import load_raw, clean, build_features
    raw = load_raw()
    df = clean(raw)
    df = build_features(df)
    return df

@st.cache_data
def load_results():
    with open(MODELS_DIR / "results.json") as f:
        return json.load(f)

@st.cache_data
def load_oof():
    return pd.read_parquet(MODELS_DIR / "oof_probs.parquet")

models_ready = (MODELS_DIR / "lgbm_fraud.txt").exists() and (MODELS_DIR / "results.json").exists()

if not models_ready:
    st.warning("Model not trained yet. Run `python train.py` first.")
    st.stop()

model, logreg, scaler = load_model()
results = load_results()
oof = load_oof()

tab_overview, tab_model, tab_threshold, tab_shap, tab_data = st.tabs([
    "📊 Overview", "📈 Model Performance", "⚖️ Threshold Tuning", "🔬 SHAP Explainability", "🗄️ Data"
])

# ── Overview ─────────────────────────────────────────────────────────────────
with tab_overview:
    r = results["lgbm"]
    d = results["data"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Transactions", f"{d['n_rows']:,}")
    col2.metric("Fraud Rate", f"{d['fraud_rate']:.3%}")
    col3.metric("AUPRC (OOF)", f"{r['auprc']:.4f}")
    col4.metric("AUROC (OOF)", f"{r['auroc']:.4f}")

    st.divider()
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Confusion Matrix")
        cm = np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]])
        fig = px.imshow(
            cm, text_auto=True,
            labels=dict(x="Predicted", y="Actual"),
            x=["Legitimate", "Fraud"], y=["Legitimate", "Fraud"],
            color_continuous_scale="Blues",
        )
        fig.update_layout(height=300, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"At threshold = {r['threshold']:.3f} (cost-optimised)")

    with col_b:
        st.subheader("Key Metrics")
        metrics_df = pd.DataFrame({
            "Metric": ["AUPRC", "AUROC", "F1", "Precision", "Recall"],
            "LightGBM (OOF)": [
                f"{r['auprc']:.4f}", f"{r['auroc']:.4f}", f"{r['f1']:.4f}",
                f"{r['precision']:.4f}", f"{r['recall']:.4f}",
            ],
            "Logistic (holdout)": [
                f"{results['logistic']['auprc']:.4f}",
                f"{results['logistic']['auroc']:.4f}",
                f"{results['logistic']['f1']:.4f}", "—", "—",
            ],
        })
        st.dataframe(metrics_df, hide_index=True, use_container_width=True)

        st.info(
            "**Why AUPRC over AUROC?** At 0.1% fraud rate, AUROC is inflated by the "
            "large negative class. AUPRC focuses on precision-recall trade-off where it matters."
        )

# ── Model Performance ─────────────────────────────────────────────────────────
with tab_model:
    y_true = oof["y_true"].values
    y_prob = oof["y_prob"].values

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Precision-Recall Curve")
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        baseline = y_true.mean()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rec, y=prec, mode="lines", name="LightGBM",
                                  line=dict(color="#2196F3", width=2)))
        fig.add_hline(y=baseline, line_dash="dash", line_color="gray",
                      annotation_text=f"Random ({baseline:.4f})")
        fig.update_layout(xaxis_title="Recall", yaxis_title="Precision",
                          height=380, margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="LightGBM",
                                  line=dict(color="#4CAF50", width=2)))
        fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                      line=dict(dash="dash", color="gray"))
        fig.update_layout(xaxis_title="False Positive Rate",
                          yaxis_title="True Positive Rate",
                          height=380, margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Score Distribution")
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=y_prob[y_true == 0], name="Legitimate",
                               nbinsx=80, opacity=0.7, marker_color="#2196F3"))
    fig.add_trace(go.Histogram(x=y_prob[y_true == 1], name="Fraud",
                               nbinsx=80, opacity=0.7, marker_color="#F44336"))
    fig.update_layout(barmode="overlay", xaxis_title="Fraud probability score",
                      yaxis_title="Count", height=320, margin=dict(t=10))
    st.plotly_chart(fig, use_container_width=True)

# ── Threshold Tuning ──────────────────────────────────────────────────────────
with tab_threshold:
    st.subheader("Business Cost Threshold Tuning")
    st.markdown(
        "Adjust the relative cost of a **false negative** (missed fraud) vs "
        "a **false positive** (false alarm). The optimal threshold minimises "
        "total expected cost."
    )

    col1, col2 = st.columns(2)
    cost_fn = col1.slider("Cost of missing fraud (FN)", 1, 50, 10)
    cost_fp = col2.slider("Cost of false alarm (FP)", 1, 10, 1)

    y_true = oof["y_true"].values
    y_prob_arr = oof["y_prob"].values

    prec, rec, thresholds = precision_recall_curve(y_true, y_prob_arr)
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos

    costs, f1s = [], []
    for t, p, r in zip(thresholds, prec[:-1], rec[:-1]):
        tp = r * n_pos
        fn = n_pos - tp
        fp = tp / (p + 1e-9) - tp
        costs.append(cost_fn * fn + cost_fp * fp)
        preds = (y_prob_arr >= t).astype(int)
        from sklearn.metrics import f1_score
        f1s.append(f1_score(y_true, preds, zero_division=0))

    best_idx = np.argmin(costs)
    best_t = thresholds[best_idx]
    best_metrics = evaluate(y_true, y_prob_arr, threshold=best_t)

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Optimal threshold", f"{best_t:.3f}")
    col_b.metric("F1 at threshold", f"{best_metrics.f1:.4f}")
    col_c.metric("Recall (fraud caught)", f"{best_metrics.recall:.4f}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=thresholds, y=costs, mode="lines",
                              name="Expected cost", line=dict(color="#FF9800")))
    fig.add_vline(x=best_t, line_dash="dash", line_color="red",
                  annotation_text=f"Optimal: {best_t:.3f}")
    fig.update_layout(xaxis_title="Decision threshold", yaxis_title="Expected cost",
                      height=350, margin=dict(t=20))
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        f"TP: {best_metrics.tp:,} · FP: {best_metrics.fp:,} · "
        f"FN: {best_metrics.fn:,} · TN: {best_metrics.tn:,}"
    )

# ── SHAP ──────────────────────────────────────────────────────────────────────
with tab_shap:
    st.subheader("SHAP Explainability")
    st.caption("Computed on a stratified 2,000-row sample for speed. Values are exact SHAP, not proxy importance.")

    @st.cache_data
    def compute_shap_sample(_model, _df, n=2000):
        rng = np.random.default_rng(42)
        fraud_idx = _df[_df[TARGET_COL] == 1].index
        legit_idx = _df[_df[TARGET_COL] == 0].index
        n_fraud = min(len(fraud_idx), n // 4)
        n_legit = n - n_fraud
        sample_idx = np.concatenate([
            rng.choice(fraud_idx, n_fraud, replace=False),
            rng.choice(legit_idx, n_legit, replace=False),
        ])
        X_sample = _df.loc[sample_idx, FEATURE_COLS]
        y_sample = _df.loc[sample_idx, TARGET_COL]
        import shap as _shap
        explainer = _shap.TreeExplainer(_model)
        shap_vals = explainer.shap_values(X_sample)
        return X_sample, y_sample, shap_vals

    with st.spinner("Computing SHAP values on sample..."):
        df_full = load_data()
        X_sample, y_sample, shap_vals = compute_shap_sample(model, df_full)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Global Feature Importance (mean |SHAP|)")
        mean_abs_shap = np.abs(shap_vals).mean(axis=0)
        shap_df = pd.DataFrame({
            "feature": FEATURE_COLS,
            "mean_abs_shap": mean_abs_shap,
        }).sort_values("mean_abs_shap", ascending=False).head(15)
        fig = px.bar(shap_df, x="mean_abs_shap", y="feature", orientation="h",
                     color="mean_abs_shap", color_continuous_scale="Reds",
                     labels={"mean_abs_shap": "Mean |SHAP|", "feature": ""})
        fig.update_layout(height=450, margin=dict(t=10), yaxis=dict(autorange="reversed"),
                          coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("SHAP Beeswarm (fraud vs legitimate)")
        fig_sw, ax = plt.subplots(figsize=(6, 5))
        import shap as _shap
        _shap.summary_plot(shap_vals, X_sample, max_display=12, show=False, plot_size=None)
        fig_sw = plt.gcf()
        fig_sw.patch.set_facecolor("none")
        st.pyplot(fig_sw, use_container_width=True)
        plt.close("all")

    st.subheader("Transaction-Level Waterfall")
    st.caption("Pick a specific transaction to see exactly why the model scored it high or low.")
    col_sel1, col_sel2 = st.columns(2)
    show_fraud = col_sel1.checkbox("Show fraud transactions only", value=True)
    sample_pool = X_sample[y_sample == 1] if show_fraud else X_sample
    tx_idx = col_sel2.selectbox("Transaction index", sample_pool.index[:50].tolist())
    local_idx = list(X_sample.index).index(tx_idx)
    fig_wf, ax2 = plt.subplots(figsize=(8, 4))
    _shap.waterfall_plot(
        _shap.Explanation(
            values=shap_vals[local_idx],
            base_values=_shap.TreeExplainer(model).expected_value,
            data=X_sample.iloc[local_idx].values,
            feature_names=FEATURE_COLS,
        ),
        show=False,
    )
    fig_wf = plt.gcf()
    st.pyplot(fig_wf, use_container_width=True)
    plt.close("all")

    st.subheader("Feature Descriptions")
    desc = {
        "orig_drained": "Sender account drained to zero — strong fraud signal",
        "dest_no_change": "Destination balance unchanged after transfer (mule pass-through)",
        "amount_to_orig_balance": "Transaction size relative to sender's balance",
        "orig_tx_count": "How many transactions the sender made in total",
        "orig_unique_dest": "Number of unique destinations (fan-out = ring indicator)",
        "orig_step_gap": "Time since sender's previous transaction (velocity burst)",
        "dest_recv_count": "How many times destination has received funds (mule re-use)",
        "is_night": "Transaction between midnight and 6am",
    }
    for feat, d_text in desc.items():
        st.markdown(f"**`{feat}`** — {d_text}")

# ── Data ──────────────────────────────────────────────────────────────────────
with tab_data:
    st.subheader("Dataset Overview")
    st.markdown(
        "**PaySim** simulates 30 days of mobile money transactions based on real "
        "transaction logs from an African mobile money provider (MTTN). Fraud only "
        "occurs in `CASH_OUT` and `TRANSFER` transaction types."
    )
    col1, col2, col3 = st.columns(3)
    d = results["data"]
    col1.metric("Rows (after filtering)", f"{d['n_rows']:,}")
    col2.metric("Fraud transactions", f"{d['n_fraud']:,}")
    col3.metric("Class imbalance", f"1 : {int((1-d['fraud_rate'])/d['fraud_rate']):,}")

    st.subheader("Graph Feature Motivation")
    st.markdown("""
    Real fraud rings operate through **money mule networks**: a single compromised
    account drains funds and fans out to many mule accounts in rapid succession.

    | Feature | What it captures |
    |---|---|
    | `orig_unique_dest` | Fan-out — one sender, many receivers |
    | `orig_step_gap` | Velocity — rapid consecutive transactions |
    | `dest_recv_count` | Mule re-use — same destination used by multiple senders |
    | `orig_drained` | Account emptied in one shot |
    | `dest_no_change` | Balance doesn't increase despite receiving funds |
    """)
