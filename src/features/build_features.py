import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any

PROCESSED_PATH = Path("data/processed/credit_data_processed.csv")
FEATURES_PATH = Path("data/processed/credit_data_features.csv")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df[df["limit_bal"] > 0].reset_index(drop=True)
    df["limit_age_ratio"] = df["limit_bal"] / (df["age"] + 1)

    bill_cols = [f"bill_amt{i}" for i in range(1, 7) if f"bill_amt{i}" in df.columns]
    pay_cols = [f"pay_amt{i}" for i in range(1, 7) if f"pay_amt{i}" in df.columns]

    df["avg_bill_amt"] = df[bill_cols].mean(axis=1)
    df["avg_pay_amt"] = df[pay_cols].mean(axis=1)

    denom = df["avg_bill_amt"]
    valid_mask = denom > 0

    df["bill_pay_ratio"] = np.nan
    if valid_mask.any():
        df.loc[valid_mask, "bill_pay_ratio"] = df.loc[valid_mask, "avg_pay_amt"] / denom[valid_mask]

    df["bill_pay_ratio"] = pd.to_numeric(df["bill_pay_ratio"], errors="coerce")

    n_neg = int((df["bill_pay_ratio"] < 0).sum())
    n_over = int((df["bill_pay_ratio"] > 5).sum())
    n_null = int(df["bill_pay_ratio"].isna().sum())

    df["bill_pay_ratio"] = df["bill_pay_ratio"].clip(lower=0, upper=5)

    n_over_after = int((df["bill_pay_ratio"] > 5).sum())
    print(f"bill_pay_ratio stats: negatives={n_neg}, >5_before_clip={n_over}, nulls={n_null}, >5_after_clip={n_over_after}")

    pay_status_cols = [f"pay_{i}" for i in range(0, 7) if f"pay_{i}" in df.columns]
    df["num_late_payments"] = (df[pay_status_cols] > 0).sum(axis=1).astype(int)
    df["max_delay"] = df[pay_status_cols].max(axis=1)
    df["avg_delay"] = df[pay_status_cols].where(df[pay_status_cols] > 0).mean(axis=1)

    def _slope(row: Any) -> float:
        y = row.values.astype(float)
        finite_mask = np.isfinite(y)
        if finite_mask.sum() >= 2:
            x = np.arange(len(y))[finite_mask]
            yv = y[finite_mask]
            try:
                return float(np.polyfit(x, yv, 1)[0])
            except Exception:
                return np.nan
        return np.nan

    df["bill_trend"] = df[bill_cols].apply(_slope, axis=1)

    df["age_bin"] = pd.cut(df["age"], bins=[0, 30, 40, 50, 60, 100], labels=False, include_lowest=True)
    for c in ["sex", "education", "marriage", "age_bin"]:
        if c in df.columns:
            df[c] = df[c].astype("category")

    print("Фичи успешно добавлены:", df.shape)
    return df


def main():
    df = pd.read_csv(PROCESSED_PATH)
    df = build_features(df)
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(FEATURES_PATH, index=False)


if __name__ == "__main__":
    main()