import json
import numpy as np
import pandas as pd
from types import SimpleNamespace
from sklearn.dummy import DummyClassifier

def test_train_model_writes_model_and_metrics(tmp_path, monkeypatch):
    import src.models.train as train

    # Поддельные данные (маленький датасет)
    X = pd.DataFrame({"f1": [0, 1, 0, 1], "f2": [1, 0, 1, 0]})
    y = pd.Series([0, 1, 0, 1])

    # Заменяем загрузчик данных
    monkeypatch.setattr(train, "load_data", lambda: (X, y))

    # Поддельный RandomizedSearchCV, возвращающий picklable estimator
    class DummySearch:
        def __init__(self, *args, **kwargs):
            self.best_params_ = {"dummy": True}
            self.best_estimator_ = None

        def fit(self, X_fit, y_fit):
            # используем sklearn DummyClassifier
            clf = DummyClassifier(strategy="most_frequent")
            clf.fit(X_fit, y_fit)
            self.best_estimator_ = clf
            return self

    monkeypatch.setattr(train, "RandomizedSearchCV", DummySearch)

    # Подмена train_test_split
    def fake_split(X_in, y_in, test_size, random_state, stratify):
        n = len(X_in)
        split = max(1, int(n * 0.5))
        return X_in.iloc[:split], X_in.iloc[split:], y_in.iloc[:split], y_in.iloc[split:]

    monkeypatch.setattr(train, "train_test_split", fake_split)

    # Перенаправляем пути сохранения в tmp
    monkeypatch.setattr(train, "MODEL_PATH", tmp_path / "model.joblib")
    monkeypatch.setattr(train, "METRICS_PATH", tmp_path / "metrics.json")
    best_model, X_test, y_test, y_prob, metrics = train.train_model(n_iter=1, cv=2)

    # Проверки: файлы созданы и содержат ожидаемые ключи
    assert (tmp_path / "model.joblib").exists()
    assert (tmp_path / "metrics.json").exists()
    loaded = json.loads((tmp_path / "metrics.json").read_text())
    assert "ROC-AUC" in loaded