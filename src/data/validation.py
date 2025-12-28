import json
import pandas as pd
import great_expectations as ge
from pathlib import Path

FEATURES_PATH = Path("data/processed/credit_data_features.csv")
EXPECTATIONS_PATH = Path("data/expectations/credit_features_suite.json")


def validate_features(raise_on_fail: bool = True):
    """
    Валидирует подготовленные признаки с использованием expectation-suite.
    """
    df = pd.read_csv(FEATURES_PATH)
    gdf = ge.from_pandas(df)

    with open(EXPECTATIONS_PATH, "r") as f:
        suite = json.load(f)

    # применяем правила
    for exp in suite["expectations"]:
        gdf.validate(expectation_suite={
            "expectations": [exp]
        })

    # полная проверка
    results = gdf.validate(expectation_suite=suite)

    if not results["success"]:
        if raise_on_fail:
            raise ValueError("Data validation failed — обнаружены аномалии")
        else:
            print("Валидация не пройдена")
    else:
        print("Validation passed successfully")

    return results


if __name__ == "__main__":
    validate_features()
