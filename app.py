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

st.set_page_config(page_title="Delhi AQI Classifier", layout="wide")

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
        "F1": f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def plot_confusion(y_true, y_pred, labels, names):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=names,
        yticklabels=names,
        cbar=False,
        annot_kws={"size": 8},
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=9)
    ax.set_ylabel("Actual", fontsize=9)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    return fig


st.title("Delhi Air Quality - AQI Category Classifier")
st.caption("BITS Pilani WILP | M.Tech AI/ML | Machine Learning Assignment")

st.sidebar.header("Controls")

uploaded = st.sidebar.file_uploader("Upload test data (CSV)", type=["csv"])

model_name = st.sidebar.selectbox(
    "Select model",
    list(MODEL_FILES.keys()),
    index=4,
)

show_all = st.sidebar.checkbox("Compare all models", value=False)

if uploaded is not None:
    data = pd.read_csv(uploaded)
    st.sidebar.caption("Uploaded: %d rows" % len(data))
else:
    data = load_default_test_data()
    if data is not None:
        st.sidebar.caption("Using test_data.csv (%d rows)" % len(data))

if data is None:
    st.warning("No data available. Please upload a CSV to continue.")
    st.stop()

if TARGET not in data.columns:
    st.error("The uploaded CSV must contain an '%s' column." % TARGET)
    st.stop()

encoder = load_encoder()
feature_cols = [c for c in data.columns if c != TARGET]

X = data[feature_cols]
y_true = encoder.transform(data[TARGET])

model = load_model(MODEL_FILES[model_name])
y_pred = model.predict(X)
y_proba = model.predict_proba(X)

st.subheader(model_name)

metrics = compute_metrics(y_true, y_pred, y_proba)
st.table(pd.DataFrame([metrics]).round(4))

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
    st.dataframe(pd.DataFrame(report).transpose().round(4))

if show_all:
    st.subheader("All Models")

    rows = []
    for name, fname in MODEL_FILES.items():
        m = load_model(fname)
        p = m.predict(X)
        pr = m.predict_proba(X)
        row = {"Model": name}
        row.update(compute_metrics(y_true, p, pr))
        rows.append(row)

    st.table(pd.DataFrame(rows).round(4))
