import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef

RANDOM_STATE = 42
DATA_FILE = "city_day.csv"

POLLUTANTS = [
    "PM2.5", "PM10", "NO", "NO2", "NOx", "NH3",
    "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene",
]
TIME_FEATURES = ["Month", "DayOfWeek", "Season"]
FEATURES = POLLUTANTS + TIME_FEATURES
TARGET = "AQI_Bucket"


def to_season(month):
    if month in (12, 1, 2):
        return 0
    if month in (3, 4, 5):
        return 1
    if month in (6, 7, 8, 9):
        return 2
    return 3


def load_data():
    df = pd.read_csv(DATA_FILE)
    d = df[df["City"] == "Delhi"].copy()
    d = d.dropna(subset=[TARGET])

    d["Date"] = pd.to_datetime(d["Date"])
    d["Month"] = d["Date"].dt.month
    d["DayOfWeek"] = d["Date"].dt.dayofweek
    d["Season"] = d["Month"].map(to_season)
    return d


def build_models():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=10, random_state=RANDOM_STATE
        ),
        "kNN": KNeighborsClassifier(n_neighbors=7),
        "Naive Bayes": GaussianNB(),
        "Random Forest (Ensemble)": RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE
        ),
    }


def evaluate(name, pipe, X_test, y_test):
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)

    auc = roc_auc_score(
        y_test, y_proba, multi_class="ovr", average="weighted"
    )
    prec = precision_score(
        y_test, y_pred, average="weighted", zero_division=0
    )
    rec = recall_score(
        y_test, y_pred, average="weighted", zero_division=0
    )
    f1 = f1_score(
        y_test, y_pred, average="weighted", zero_division=0
    )

    return {
        "ML Model Name": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": auc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "MCC": matthews_corrcoef(y_test, y_pred),
    }


def main():
    os.makedirs("model", exist_ok=True)

    d = load_data()

    X = d[FEATURES]
    le = LabelEncoder()
    y = le.fit_transform(d[TARGET])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("Features :", len(FEATURES))
    print("Rows     :", len(d))
    print("Classes  :", list(le.classes_))
    print("Train    :", X_train.shape, " Test:", X_test.shape)
    print("Test class counts:", np.bincount(y_test))
    print("-" * 62)

    rows = []
    for name, clf in build_models().items():
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", clf),
        ])
        pipe.fit(X_train, y_train)

        rows.append(evaluate(name, pipe, X_test, y_test))

        fname = name.split(" (")[0].replace(" ", "_").lower()
        joblib.dump(pipe, "model/" + fname + ".pkl")
        print("saved  model/" + fname + ".pkl")

    results = pd.DataFrame(rows).round(4)

    joblib.dump(le, "model/label_encoder.pkl")

    test_df = X_test.copy()
    test_df[TARGET] = le.inverse_transform(y_test)
    test_df.to_csv("test_data.csv", index=False)

    results.to_csv("model/metrics_summary.csv", index=False)

    print("-" * 62)
    print(results.to_string(index=False))
    print("-" * 62)
    print("test_data.csv :", test_df.shape)
    print("model/        :", sorted(os.listdir("model")))
    print()
    print("=== Markdown table for README ===")
    print(results.to_markdown(index=False))


if __name__ == "__main__":
    main()
