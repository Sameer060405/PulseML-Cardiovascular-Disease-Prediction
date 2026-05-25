"""Interactive PulseML risk prediction dashboard."""

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import auc, confusion_matrix, roc_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import BEST_MODEL_PATH, METRICS_PATH  # noqa: E402
from src.data_loader import load_dataset  # noqa: E402
from src.predict import predict  # noqa: E402
from src.preprocess import split_data  # noqa: E402


def render_model_comparison(metrics: dict) -> None:
    df = pd.DataFrame(metrics["models"])
    df_melt = df.melt(
        id_vars=["model"],
        value_vars=["accuracy", "precision", "recall", "f1", "roc_auc"],
        var_name="metric",
        value_name="score",
    )
    fig = px.bar(
        df_melt,
        x="model",
        y="score",
        color="metric",
        barmode="group",
        title="Tuned model performance (test set)",
        labels={"model": "Model", "score": "Score"},
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(yaxis_range=[0, 1.05], xaxis_tickangle=-20, height=400)
    st.plotly_chart(fig, use_container_width=True)

    cv_df = df.sort_values("cv_roc_auc_mean", ascending=True)
    fig_cv = px.bar(
        cv_df,
        x="cv_roc_auc_mean",
        y="model",
        orientation="h",
        title="5-fold cross-validation ROC-AUC (full train set)",
        labels={"cv_roc_auc_mean": "Mean ROC-AUC", "model": "Model"},
        color="cv_roc_auc_mean",
        color_continuous_scale="Teal",
    )
    fig_cv.update_layout(height=320, showlegend=False)
    st.plotly_chart(fig_cv, use_container_width=True)


def render_roc_and_confusion() -> None:
    pipeline = joblib.load(BEST_MODEL_PATH)
    _, X_test, _, y_test = split_data(load_dataset())

    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = pipeline.predict(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    fig_roc = go.Figure()
    fig_roc.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"Best model (AUC = {roc_auc:.3f})",
            line=dict(color="#E45756", width=2),
        )
    )
    fig_roc.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random",
            line=dict(color="#888", dash="dash"),
        )
    )
    fig_roc.update_layout(
        title="ROC curve — deployed model",
        xaxis_title="False positive rate",
        yaxis_title="True positive rate",
        height=380,
        legend=dict(yanchor="bottom", y=0.02, xanchor="right", x=0.99),
    )

    cm = confusion_matrix(y_test, y_pred)
    fig_cm = px.imshow(
        cm,
        text_auto=True,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=["No disease", "Disease"],
        y=["No disease", "Disease"],
        title="Confusion matrix — deployed model",
        color_continuous_scale="Blues",
    )
    fig_cm.update_layout(height=380)

    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(fig_roc, use_container_width=True)
    with col_b:
        st.plotly_chart(fig_cm, use_container_width=True)


st.set_page_config(
    page_title="PulseML | Cardiovascular Risk",
    page_icon="❤️",
    layout="wide",
)

st.title("PulseML — Cardiovascular Disease Prediction")
st.caption(
    "Tuned ML models on ~70,000 real patient records (public cardiovascular disease dataset)."
)

if not BEST_MODEL_PATH.exists():
    st.error("Trained models not found. Run: `python run_pipeline.py` from the project root.")
    st.stop()

metrics = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else None
if metrics:
    st.info(
        f"Dataset: **{metrics.get('dataset_rows', 'N/A'):,}** patients · "
        f"Best model: **{metrics.get('best_model', 'N/A')}** · "
        f"ROC-AUC: **{max(m['roc_auc'] for m in metrics['models']):.1%}**"
    )

tab_predict, tab_eval = st.tabs(["Predict", "Model evaluation"])

with tab_predict:
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("Patient profile")
        age_years = st.slider("Age (years)", 29, 65, 50)
        gender = st.selectbox("Gender", [1, 2], format_func=lambda x: "Female" if x == 1 else "Male")
        height = st.slider("Height (cm)", 140, 220, 170)
        weight = st.slider("Weight (kg)", 40, 200, 75)
        bmi = round(weight / (height / 100) ** 2, 1)
        st.metric("BMI (computed)", bmi)
        ap_hi = st.slider("Systolic BP (ap_hi)", 90, 250, 120)
        ap_lo = st.slider("Diastolic BP (ap_lo)", 60, 150, 80)
        cholesterol = st.selectbox(
            "Cholesterol",
            [1, 2, 3],
            format_func=lambda x: ["Normal", "Above normal", "Well above normal"][x - 1],
        )
        gluc = st.selectbox(
            "Glucose",
            [1, 2, 3],
            format_func=lambda x: ["Normal", "Above normal", "Well above normal"][x - 1],
        )
        smoke = st.selectbox("Smoker", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        alco = st.selectbox("Alcohol intake", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        active = st.selectbox("Physically active", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

        patient = {
            "age_years": age_years,
            "gender": gender,
            "height": height,
            "weight": weight,
            "bmi": bmi,
            "ap_hi": ap_hi,
            "ap_lo": ap_lo,
            "cholesterol": cholesterol,
            "gluc": gluc,
            "smoke": smoke,
            "alco": alco,
            "active": active,
        }

        if st.button("Predict risk", type="primary"):
            st.session_state["result"] = predict(patient)

    with col_right:
        st.subheader("Prediction")
        if "result" in st.session_state:
            r = st.session_state["result"]
            color = "#E45756" if r["risk_level"] == "High" else "#54A24B"
            st.markdown(
                f"""
                <div style="padding:24px;border-radius:12px;background:{color}22;border:2px solid {color}">
                <h2 style="color:{color};margin:0">{r['risk_level']} Risk</h2>
                <p style="font-size:1.2rem;margin:8px 0 0">Probability: <b>{r['probability']:.1%}</b></p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            gauge = px.pie(
                values=[r["probability"], 1 - r["probability"]],
                names=["Disease", "No disease"],
                hole=0.65,
                color_discrete_sequence=["#E45756", "#54A24B"],
            )
            gauge.update_layout(showlegend=False, height=280, margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(gauge, use_container_width=True)
        else:
            st.info("Enter patient data and click **Predict risk**.")

        if metrics:
            st.subheader("Tuned model leaderboard")
            df_metrics = pd.DataFrame(metrics["models"])
            st.dataframe(
                df_metrics[
                    ["model", "accuracy", "precision", "recall", "f1", "roc_auc", "cv_roc_auc_mean"]
                ].style.format(
                    {
                        "accuracy": "{:.2%}",
                        "precision": "{:.2%}",
                        "recall": "{:.2%}",
                        "f1": "{:.2%}",
                        "roc_auc": "{:.2%}",
                        "cv_roc_auc_mean": "{:.2%}",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

with tab_eval:
    st.subheader("Model evaluation")
    if metrics:
        render_model_comparison(metrics)
        st.divider()
        st.subheader("Deployed model — test set")
        render_roc_and_confusion()
    else:
        st.warning("Metrics not found. Run `python run_pipeline.py` first.")
