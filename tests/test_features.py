import pandas as pd
from src.features.build_features import build_features

def test_build_features_basic():
    df = pd.read_csv("data/processed/credit_data_processed.csv")
    f = build_features(df)

    assert (f["limit_bal"] > 0).all()

    ratios = f["bill_pay_ratio"].dropna()
    assert (ratios >= 0).all(), "Есть отрицательные bill_pay_ratio"
    assert (ratios <= 5).all(), "Есть bill_pay_ratio > 5"

    assert "age_bin" in f.columns
    assert hasattr(f["age_bin"].dtype, "name")

    assert (~f["bill_trend"].isin([float("inf"), float("-inf")])).all()