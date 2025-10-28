import pandas as pd
import numpy as np
from pathlib import Path

PROCESSED_PATH = Path("data/processed/credit_data_processed.csv")
FEATURES_PATH = Path("data/processed/credit_data_features.csv")

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Финансовые соотношения
    df["limit_age_ratio"] = df["limit_bal"] / (df["age"] + 1)
    df["avg_bill_amt"] = df[[f"bill_amt{i}" for i in range(1, 7)]].mean(axis=1)
    df["avg_pay_amt"] = df[[f"pay_amt{i}" for i in range(1, 7)]].mean(axis=1)
    df["bill_pay_ratio"] = df["avg_pay_amt"] / (df["avg_bill_amt"] + 1)
    
    # История задержек платежей
    pay_cols = [f"pay_{i}" for i in range(0, 7) if f"pay_{i}" in df.columns]
    df["num_late_payments"] = (df[pay_cols] > 0).sum(axis=1)
    df["max_delay"] = df[pay_cols].max(axis=1)
    df["avg_delay"] = df[pay_cols].mean(axis=1)

    # Тренд по задолженности
    bill_cols = [f"bill_amt{i}" for i in range(1, 7)]
    df["bill_trend"] = df[bill_cols].apply(lambda x: np.polyfit(range(len(x)), x, 1)[0], axis=1)

    # Биннинг возраста
    df["age_bin"] = pd.cut(df["age"], bins=[0, 30, 40, 50, 60, 100], labels=False)
    
    # Удаляем выбросы и невозможные значения
    df = df[df["limit_bal"] > 0]

    print("Фичи успешно добавлены:", df.shape)
    return df

def main():
    df = pd.read_csv(PROCESSED_PATH)
    df = build_features(df)
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(FEATURES_PATH, index=False)
    

if __name__ == "__main__":
    main()