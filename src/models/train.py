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
import numpy as np
import joblib
from src.models.plots import plot_roc_curve
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt

FEATURES_PATH = Path("data/processed/credit_data_features.csv")
MODEL_PATH = Path("models/random_forest_pd.pkl")


def load_data():
    df = pd.read_csv(FEATURES_PATH)
    y = df["default_payment_next_month"]
    X = df.drop(columns=["default_payment_next_month"])
    return X, y


def build_preprocessor(X):
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features)
        ]
    )

    return preprocessor



def train_model():

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
            n_iter=20,
            scoring="roc_auc",
            cv=5,
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
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1": f1_score(y_test, y_pred)
        }

        print("Classification report:")
        print(classification_report(y_test, y_pred))

        print("Metrics on test set:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
            
        mlflow.log_metrics(metrics)
        
        RocCurveDisplay.from_predictions(y_test, y_prob)
        plt.title("ROC curve")
        plt.savefig("roc_curve.png")
        plt.close()
        
        mlflow.log_artifact("roc_curve.png")
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model"
        )

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_model, MODEL_PATH)

        print(f"Модель сохранена: {MODEL_PATH}")

        return best_model, X_test, y_test, y_prob

if __name__ == "__main__":
    best_model, X_test, y_test, y_prob = train_model()
    plot_roc_curve(y_test, y_prob)