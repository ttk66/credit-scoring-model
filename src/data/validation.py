import json
import pandas as pd
import great_expectations as ge
from pathlib import Path

FEATURES_PATH = Path("data/processed/credit_data_features.csv")
EXPECTATIONS_PATH = Path("data/expectations/expectations.json")


def _simple_validate(df: pd.DataFrame, suite: dict):
    results_list = []
    all_ok = True

    for exp in suite.get("expectations", []):
        etype = exp.get("expectation_type")
        kw = exp.get("kwargs", {}) or {}
        success = True
        info = {}

        try:
            if etype in ("expect_table_row_count_to_be_between", "expect_table_row_count_to_be_greater_than"):
                min_v = kw.get("min_value")
                max_v = kw.get("max_value")
                rc = len(df)
                if min_v is not None and rc < min_v:
                    success = False
                if max_v is not None and rc > max_v:
                    success = False
                info["observed_value"] = rc

            elif etype in ("expect_table_columns_to_match_set", "expect_table_columns_to_contain_set"):
                required = set(kw.get("column_set", []))
                dfcols = set(df.columns.tolist())
                missing = sorted(list(required - dfcols))
                if missing:
                    success = False
                info["missing_columns"] = missing

            elif etype == "expect_column_values_to_be_between":
                col = kw.get("column")
                if col not in df.columns:
                    success = False
                    info["reason"] = "column_not_found"
                else:
                    ser = df[col].dropna()
                    min_v = kw.get("min_value")
                    max_v = kw.get("max_value")
                    if min_v is not None and (ser < min_v).any():
                        success = False
                    if max_v is not None and (ser > max_v).any():
                        success = False
                    info["observed_min"] = float(ser.min()) if len(ser) > 0 else None
                    info["observed_max"] = float(ser.max()) if len(ser) > 0 else None

            elif etype == "expect_column_values_to_not_be_null":
                col = kw.get("column")
                if col not in df.columns:
                    success = False
                    info["reason"] = "column_not_found"
                else:
                    nulls = int(df[col].isnull().sum())
                    if nulls > 0:
                        success = False
                    info["null_count"] = nulls

            elif etype == "expect_column_values_to_be_in_set":
                col = kw.get("column")
                allowed = set(kw.get("value_set", []))
                if col not in df.columns:
                    success = False
                    info["reason"] = "column_not_found"
                else:
                    ser = df[col].dropna()
                    bad = ser[~ser.isin(allowed)]
                    if len(bad) > 0:
                        success = False
                    info["unexpected_values"] = sorted(list(set(bad.tolist())))

            else:
                success = False
                info["reason"] = "unknown_expectation"

        except Exception as e:
            success = False
            info["exception"] = str(e)

        results_list.append({"expectation_type": etype, "success": success, "result": info})
        if not success:
            all_ok = False

    return {"success": all_ok, "results": results_list}


def validate_features(raise_on_fail: bool = True):
    """
    Валидирует подготовленные признаки с использованием expectation-suite.
    В случаях, когда установлен Great Expectations старой/неподдерживаемой версии,
    используется локальный простой валидатор (fallback).
    """
    df = pd.read_csv(FEATURES_PATH)

    if not EXPECTATIONS_PATH.exists():
        raise FileNotFoundError(f"Expectations file not found at {EXPECTATIONS_PATH}")

    with open(EXPECTATIONS_PATH, "r", encoding="utf-8") as f:
        suite = json.load(f)

    try:
        gdf = ge.from_pandas(df)
    except Exception:
        # Fallback на простой валидатор
        results = _simple_validate(df, suite)
        if not results["success"]:
            if raise_on_fail:
                raise ValueError("Data validation failed — обнаружены аномалии (simple validator)")
            else:
                print("Валидация не пройдена (simple validator)")
        else:
            print("Validation passed successfully (simple validator)")
        return results

    # Если ge.from_pandas сработал, применяем ожидания через GE
    for exp in suite["expectations"]:
        gdf.validate(expectation_suite={"expectations": [exp]})

    results = gdf.validate(expectation_suite=suite)

    if not results.get("success", False):
        if raise_on_fail:
            raise ValueError("Data validation failed — обнаружены аномалии")
        else:
            print("Валидация не пройдена")
    else:
        print("Validation passed successfully")

    return results


if __name__ == "__main__":
    validate_features()
