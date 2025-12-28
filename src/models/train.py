import json
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score, classification_report, RocCurveDisplay
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
import joblib
from src.models.plots import plot_roc_curve
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
from sklearn import __version__ as _sklearn_version

FEATURES_PATH = Path("data/processed/credit_data_features.csv")
MODEL_PATH = Path("models/best_model.joblib")
METRICS_PATH = Path("models/model_metrics.json")


def load_data():
    df = pd.read_csv(FEATURES_PATH)
    df = df.drop(columns=["id"], errors="ignore")
    target = "default_payment_next_month"
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found in {FEATURES_PATH}")

    categorical_candidates = ["sex", "education", "marriage", "age_bin"]
    present_cat = [c for c in categorical_candidates if c in df.columns]
    for c in present_cat:
        df[c] = df[c].astype("category")

    y = df[target]
    X = df.drop(columns=[target])
    return X, y


def build_preprocessor(X):
    explicit_cat = [c for c in ["sex", "education", "marriage", "age_bin"] if c in X.columns]
    if explicit_cat:
        categorical_features = explicit_cat
    else:
        categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

    numeric_features = [c for c in X.columns if c not in categorical_features]

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    sk_major, sk_minor = map(int, _sklearn_version.split(".")[:2])
    if (sk_major, sk_minor) >= (1, 2):
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    else:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", ohe)
    ])

    transformers = []
    if numeric_features:
        transformers.append(("num", numeric_pipeline, numeric_features))
    if categorical_features:
        transformers.append(("cat", categorical_pipeline, categorical_features))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    return preprocessor


def train_model(n_iter: int = 20, cv: int = 5):

    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    preprocessor = build_preprocessor(X)

    model = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight="balanced")

    pipe = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model)
    ])

    param_grid = {
        "model__n_estimators": [200, 300, 400, 500],
        "model__max_depth": [None, 6, 8, 10, 12],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", "log2"]
    }

    with mlflow.start_run():

        search = RandomizedSearchCV(
            pipe,
            param_distributions=param_grid,
            n_iter=n_iter,
            scoring="roc_auc",
            cv=cv,
            verbose=2,
            n_jobs=-1,
            random_state=42
        )

        search.fit(X_train, y_train)

        print("Best params:")
        print(search.best_params_)

        best_model = search.best_estimator_
        mlflow.log_params(search.best_params_)

        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]

        metrics = {
            "ROC-AUC": roc_auc_score(y_test, y_prob),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1": f1_score(y_test, y_pred, zero_division=0)
        }

        print("Classification report:")
        print(classification_report(y_test, y_pred))

        print("Metrics on test set:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

        mlflow.log_metrics(metrics)

        METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
        metrics_json = {k: float(v) for k, v in metrics.items()}
        with open(METRICS_PATH, "w", encoding="utf-8") as f:
            json.dump(metrics_json, f, ensure_ascii=False, indent=2)
        mlflow.log_artifact(str(METRICS_PATH))

        RocCurveDisplay.from_predictions(y_test, y_prob)
        plt.title("ROC curve")
        plt.savefig("roc_curve.png")
        plt.close()

        mlflow.log_artifact("roc_curve.png")
        mlflow.sklearn.log_model(
            sk_model=best_model,
            name="model"
        )

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_model, MODEL_PATH)

        print(f"Модель сохранена: {MODEL_PATH}")

        return best_model, X_test, y_test, y_prob, metrics


if __name__ == "__main__":
    best_model, X_test, y_test, y_prob, metrics = train_model()
    plot_roc_curve(y_test, y_prob)