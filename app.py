from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    average_precision_score,
)

ARTIFACTS = Path("artifacts")
MODEL_PATH = ARTIFACTS / "best_model.joblib"
META_PATH = ARTIFACTS / "metadata.json"

st.set_page_config(
    page_title="Predictive Maintenance AI",
    page_icon="⚙️",
    layout="wide",
)

st.title("⚙️ Intelligent Predictive Maintenance")
st.write(
    "Machine failure prediction using sensor measurements "
    "and machine learning."
)

if not MODEL_PATH.exists() or not META_PATH.exists():
    st.error("Model not found. Run `python train.py` first.")
    st.stop()

model = joblib.load(MODEL_PATH)
metadata = json.loads(META_PATH.read_text())

FEATURES = metadata["features"]
NUMERIC = metadata["numeric_features"]
THRESHOLD = float(metadata["threshold"])

st.sidebar.header("Model information")
st.sidebar.write("Selected algorithm:", metadata["model_name"])
st.sidebar.write("Alert threshold:", round(THRESHOLD, 4))
st.sidebar.caption(
    "Educational prototype using synthetic dataset records."
)

tab1, tab2, tab3 = st.tabs([
    "Single prediction",
    "Batch CSV prediction",
    "Model evaluation",
])

# --------------------------------------------------
# TAB 1: Manual sensor input
# --------------------------------------------------

with tab1:
    st.subheader("Enter current machine measurements")

    with st.form("sensor_form"):
        col1, col2 = st.columns(2)

        with col1:
            product_type = st.selectbox(
                "Product type", ["L", "M", "H"]
            )
            air_temp = st.number_input(
                "Air temperature (K)",
                min_value=250.0,
                max_value=350.0,
                value=300.0,
                step=0.1,
            )
            process_temp = st.number_input(
                "Process temperature (K)",
                min_value=250.0,
                max_value=400.0,
                value=310.0,
                step=0.1,
            )

        with col2:
            rpm = st.number_input(
                "Rotational speed (rpm)",
                min_value=0,
                max_value=10000,
                value=1500,
                step=10,
            )
            torque = st.number_input(
                "Torque (Nm)",
                min_value=0.0,
                max_value=200.0,
                value=40.0,
                step=0.5,
            )
            tool_wear = st.number_input(
                "Tool wear (min)",
                min_value=0,
                max_value=500,
                value=100,
                step=1,
            )

        submitted = st.form_submit_button(
            "Predict machine condition",
            type="primary",
        )

    if submitted:
        sample = pd.DataFrame([{
            "Type": product_type,
            "Air temperature [K]": air_temp,
            "Process temperature [K]": process_temp,
            "Rotational speed [rpm]": rpm,
            "Torque [Nm]": torque,
            "Tool wear [min]": tool_wear,
        }])[FEATURES]

        score = float(model.predict_proba(sample)[0, 1])
        failed = score >= THRESHOLD

        st.subheader("Prediction result")

        c1, c2 = st.columns(2)
        c1.metric("Model score", f"{score:.1%}")
        c2.metric(
            "Decision",
            "Flag for inspection" if failed else "Not flagged"
        )

        st.progress(min(max(score, 0.0), 1.0))

        if failed:
            st.warning(
                "The model flagged this operating condition. "
                "Inspect the machine and verify the sensor readings."
            )
        else:
            st.success(
                "The model did not flag this operating condition. "
                "This does not guarantee that the machine is safe."
            )

        st.caption(
            "The score is a model output, not necessarily a calibrated "
            "real-world probability or a prediction of future failure."
        )

# --------------------------------------------------
# TAB 2: Batch predictions from CSV
# --------------------------------------------------

with tab2:
    st.subheader("Upload sensor records")
    st.write(
        "Upload a CSV containing the five numerical measurements "
        "and the Type column. Extra columns are ignored."
    )

    uploaded = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        key="batch_csv",
    )

    if uploaded is not None:
        try:
            data = pd.read_csv(uploaded)
            data.columns = data.columns.str.strip()

            missing = [
                col for col in FEATURES if col not in data.columns
            ]

            if missing:
                st.error(f"Missing required columns: {missing}")
            elif data.empty:
                st.error("The uploaded CSV has no records.")
            else:
                X = data[FEATURES].copy()

                for col in NUMERIC:
                    X[col] = pd.to_numeric(
                        X[col], errors="coerce"
                    )

                X["Type"] = X["Type"].astype("string").str.strip()

                if not set(X["Type"].dropna().unique()).issubset(
                    {"L", "M", "H"}
                ):
                    st.error("Type must contain L, M, or H.")
                elif np.isinf(
                    X[NUMERIC].to_numpy(dtype=float)
                ).any():
                    st.error("Numeric features contain infinity.")
                else:
                    scores = model.predict_proba(X)[:, 1]
                    preds = (scores >= THRESHOLD).astype(int)

                    output = data.copy()
                    output["failure_score"] = scores
                    output["predicted_failure"] = preds
                    output["status"] = np.where(
                        preds == 1,
                        "Flag for inspection",
                        "Not flagged",
                    )

                    a, b, c = st.columns(3)
                    a.metric("Records", len(output))
                    b.metric("Flagged", int(preds.sum()))
                    c.metric(
                        "Flag rate",
                        f"{preds.mean() * 100:.2f}%"
                    )

                    st.dataframe(
                        output.sort_values(
                            "failure_score", ascending=False
                        ),
                        use_container_width=True,
                    )

                    st.download_button(
                        "Download predictions",
                        data=output.to_csv(
                            index=False
                        ).encode("utf-8"),
                        file_name="maintenance_predictions.csv",
                        mime="text/csv",
                    )

        except Exception as exc:
            st.error(f"Could not process CSV: {exc}")

# --------------------------------------------------
# TAB 3: Evaluation and feature importance
# --------------------------------------------------

with tab3:
    st.subheader("Held-out test results")

    metrics = metadata["test_metrics"]
    cols = st.columns(4)

    cols[0].metric("Precision", f"{metrics['precision']:.3f}")
    cols[1].metric("Recall", f"{metrics['recall']:.3f}")
    cols[2].metric("F1-score", f"{metrics['f1']:.3f}")
    cols[3].metric(
        "Average precision",
        f"{metrics['average_precision']:.3f}"
    )

    st.write("Selected model:", metadata["model_name"])
    st.write("ROC AUC:", round(metrics["roc_auc"], 4))
    st.write("Accuracy:", round(metrics["accuracy"], 4))

    cm = np.array(metrics["confusion_matrix"])

    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No failure", "Failure"],
    ).plot(ax=ax, colorbar=False)
    ax.set_title("Test-set confusion matrix")
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "Rows are actual labels; columns are predicted labels. "
        "A false negative is a failure that the model missed."
    )

    comparison = metadata["validation_comparison"]
    comparison_df = pd.DataFrame(comparison).T
    st.subheader("Validation model comparison")
    st.dataframe(comparison_df, use_container_width=True)

    importance_path = ARTIFACTS / "feature_importance.csv"

    if importance_path.exists():
        importance = pd.read_csv(importance_path).head(10)

        st.subheader("Most important model features")
        fig, ax = plt.subplots()
        ax.barh(
            importance["feature"][::-1],
            importance["importance"][::-1],
            color="teal",
        )
        ax.set_xlabel("Feature importance")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.caption(
        "Feature importance indicates predictive contribution, "
        "not causation. Results apply to this synthetic dataset."
    )