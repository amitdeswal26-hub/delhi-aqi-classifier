import os

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

st.set_page_config(
    page_title="Delhi AQI Classifier",
    layout="wide",
)

MODEL_DIR = "model"
TARGET = "AQI_Bucket"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest (Ensemble)": "random_forest.pkl",
}


@st.cache_resource
def load_model(filename):
    return joblib.load(os.path.join(MODEL_DIR, filename))


@st.cache_resource
def load_encoder():
    return joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))


@st.cache_data
def load_default_test_data():
    if os.path.exists("test_data.csv"):
        return pd.read_csv("test_data.csv")
    return None


def compute_metrics(y_true, y_pred, y_proba):
    try:
        auc = roc_auc_score(
            y_true, y_proba, multi_class="ovr", average="weighted"
        )
    except ValueError:
        auc = np.nan

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "Recall": recall_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "F1 Score": f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def plot_confusion(y_true, y_pred, labels, names):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=names,
        yticklabels=names,
        cbar=False,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    return fig


st.title("Delhi Air Quality - AQI Category Classifier")
st.caption("BITS Pilani WILP | M.Tech AI/ML | Machine Learning Assignment")

st.markdown(
    "Predicts the CPCB **AQI category** "
    "(Good / Satisfactory / Moderate / Poor / Very Poor / Severe) "
    "from 12 pollutant concentrations plus 3 derived time features. "
    "The raw `AQI` value is excluded from the feature set to avoid "
    "target leakage."
)

st.sidebar.header("Controls")

uploaded = st.sidebar.file_uploader(
    "Upload test data (CSV)",
    type=["csv"],
    help="Upload test_data.csv from the repository, or your own file "
         "with the same columns.",
)

model_name = st.sidebar.selectbox(
    "Select model",
    list(MODEL_FILES.keys()),
    index=4,
)

show_all = st.sidebar.checkbox("Compare all models", value=False)

if uploaded is not None:
    data = pd.read_csv(uploaded)
    st.sidebar.success("Uploaded: %d rows" % len(data))
else:
    data = load_default_test_data()
    if data is not None:
        st.sidebar.info("Using bundled test_data.csv (%d rows)" % len(data))

if data is None:
    st.warning("No data available. Please upload a CSV to continue.")
    st.stop()

if TARGET not in data.columns:
    st.error(
        "The uploaded CSV must contain a '%s' column with the true "
        "labels, so that evaluation metrics can be computed." % TARGET
    )
    st.stop()

encoder = load_encoder()
feature_cols = [c for c in data.columns if c != TARGET]

X = data[feature_cols]
y_true = encoder.transform(data[TARGET])

with st.expander("Preview input data", expanded=False):
    st.dataframe(data.head(20), use_container_width=True)
    st.write("Shape:", data.shape)

st.subheader("Results - %s" % model_name)

model = load_model(MODEL_FILES[model_name])
y_pred = model.predict(X)
y_proba = model.predict_proba(X)

metrics = compute_metrics(y_true, y_pred, y_proba)

cols = st.columns(6)
for col, (label, value) in zip(cols, metrics.items()):
    col.metric(label, "n/a" if pd.isna(value) else "%.4f" % value)

left, right = st.columns([1, 1])

with left:
    st.markdown("**Confusion Matrix**")
    fig = plot_confusion(
        y_true,
        y_pred,
        labels=list(range(len(encoder.classes_))),
        names=list(encoder.classes_),
    )
    st.pyplot(fig)

with right:
    st.markdown("**Classification Report**")
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(encoder.classes_))),
        target_names=list(encoder.classes_),
        zero_division=0,
        output_dict=True,
    )
    st.dataframe(
        pd.DataFrame(report).transpose().round(4),
        use_container_width=True,
    )

with st.expander("Predictions", expanded=False):
    out = data.copy()
    out["Predicted"] = encoder.inverse_transform(y_pred)
    out["Correct"] = out["Predicted"] == out[TARGET]
    st.dataframe(out.head(100), use_container_width=True)
    st.download_button(
        "Download predictions as CSV",
        out.to_csv(index=False).encode("utf-8"),
        file_name="predictions.csv",
        mime="text/csv",
    )

if show_all:
    st.subheader("All Models - Comparison")

    rows = []
    for name, fname in MODEL_FILES.items():
        m = load_model(fname)
        p = m.predict(X)
        pr = m.predict_proba(X)
        row = {"ML Model Name": name}
        row.update(compute_metrics(y_true, p, pr))
        rows.append(row)

    comparison = pd.DataFrame(rows).round(4)
    st.dataframe(comparison, use_container_width=True, hide_index=True)

    best = comparison.loc[comparison["Accuracy"].idxmax(), "ML Model Name"]
    st.success("Best performing model on this data: **%s**" % best)

    chart = comparison.set_index("ML Model Name")[
        ["Accuracy", "Precision", "Recall", "F1 Score"]
    ]
    st.bar_chart(chart)

st.divider()
st.caption(
    "Dataset: Air Quality Data in India (2015-2020), Kaggle - "
    "city_day.csv filtered to Delhi. Source: CPCB."
)
