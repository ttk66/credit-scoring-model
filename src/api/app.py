from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

MODEL_PATH = Path("models/best_model.joblib")

app = FastAPI(
    title="Credit Scoring Model API",
    description="REST API for default prediction",
    version="1.0"
)

class RawFeatures(BaseModel):
    limit_bal: float
    age: float
    sex: int
    education: int
    marriage: int
    pay_0: int
    pay_2: int
    pay_3: int
    pay_4: int
    pay_5: int
    pay_6: int
    bill_amt1: float
    bill_amt2: float
    bill_amt3: float
    bill_amt4: float
    bill_amt5: float
    bill_amt6: float
    pay_amt1: float
    pay_amt2: float
    pay_amt3: float
    pay_amt4: float
    pay_amt5: float
    pay_amt6: float

def build_features_api(df: pd.DataFrame) -> pd.DataFrame:
    """Адаптированная версия build_features для API"""
    df = df.copy()
    
    mask = df["limit_bal"] > 0
    df["limit_age_ratio"] = np.where(
        mask,
        df["limit_bal"] / (df["age"] + 1),
        0 
    )

    bill_cols = [f"bill_amt{i}" for i in range(1, 7)]
    pay_cols = [f"pay_amt{i}" for i in range(1, 7)]
    
    df["avg_bill_amt"] = df[bill_cols].mean(axis=1)
    df["avg_pay_amt"] = df[pay_cols].mean(axis=1)

    denom = df["avg_bill_amt"]
    valid_mask = denom > 0
    
    df["bill_pay_ratio"] = np.nan
    if valid_mask.any():
        df.loc[valid_mask, "bill_pay_ratio"] = df.loc[valid_mask, "avg_pay_amt"] / denom[valid_mask]
    
    df["bill_pay_ratio"] = pd.to_numeric(df["bill_pay_ratio"], errors="coerce")
    df["bill_pay_ratio"] = df["bill_pay_ratio"].fillna(0)
    df["bill_pay_ratio"] = df["bill_pay_ratio"].clip(lower=0, upper=5)

    # Добавляем pay_1 если его нет
    if 'pay_1' not in df.columns:
        df['pay_1'] = 0
    
    pay_status_cols = [f"pay_{i}" for i in range(0, 7)]
    df["num_late_payments"] = (df[pay_status_cols] > 0).sum(axis=1).astype(int)
    df["max_delay"] = df[pay_status_cols].max(axis=1)
    
    df_positive = df[pay_status_cols].where(df[pay_status_cols] > 0)
    df["avg_delay"] = df_positive.mean(axis=1)
    df["avg_delay"] = df["avg_delay"].fillna(0)

    # Тренд счетов
    def _slope(row):
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
    df["bill_trend"] = df["bill_trend"].fillna(0)

    # Биннинг возраста
    bins = [0, 30, 40, 50, 60, 100]
    labels = [0, 1, 2, 3, 4]
    
    df["age_bin"] = pd.cut(df["age"], bins=bins, labels=labels, include_lowest=True)
    df["age_bin"] = df["age_bin"].cat.add_categories([-1]).fillna(-1)
    
    # Категориальные фичи
    for c in ["sex", "education", "marriage", "age_bin"]:
        df[c] = df[c].astype("category")
    
    return df

@app.on_event("startup")
def load_model():
    global model, expected_features
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model not found: {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)
    print("Model loaded")
    
    if hasattr(model, 'feature_names_in_'):
        expected_features = model.feature_names_in_.tolist()
        print(f"Model expects {len(expected_features)} features")
    else:
        expected_features = [
            "limit_bal", "sex", "education", "marriage", "age",
            "pay_0", "pay_2", "pay_3", "pay_4", "pay_5", "pay_6",
            "bill_amt1", "bill_amt2", "bill_amt3", "bill_amt4", "bill_amt5", "bill_amt6",
            "pay_amt1", "pay_amt2", "pay_amt3", "pay_amt4", "pay_amt5", "pay_amt6",
            "limit_age_ratio", "avg_bill_amt", "avg_pay_amt", "bill_pay_ratio",
            "num_late_payments", "max_delay", "avg_delay", "bill_trend", "age_bin"
        ]

@app.post("/predict")
def predict(features: RawFeatures):
    try:
        raw_data = features.dict()
        raw_df = pd.DataFrame([raw_data])
        
        print(f"Received {len(raw_data)} raw features")
        
        processed_df = build_features_api(raw_df)
        
        print(f"After feature engineering: {processed_df.shape}")
        
        missing_features = set(expected_features) - set(processed_df.columns)
        if missing_features:
            print(f"Missing features: {missing_features}")
            for feat in missing_features:
                processed_df[feat] = 0
        
        processed_df = processed_df[expected_features]
        proba = model.predict_proba(processed_df)[0, 1]
        pred = int(proba >= 0.5)
        
        return {
            "prediction": pred,
            "probability": float(proba),
            "features_processed": processed_df.shape[1]
        }
        
    except Exception as e:
        import traceback
        error_details = f"{str(e)}\n\n{traceback.format_exc()}"
        raise HTTPException(status_code=400, detail=error_details)

@app.get("/model_info")
def get_model_info():
    """Информация о модели"""
    return {
        "model_type": "Pipeline",
        "pipeline_steps": list(model.named_steps.keys()) if hasattr(model, 'named_steps') else [],
        "expected_features": expected_features if 'expected_features' in globals() else [],
        "expected_features_count": len(expected_features) if 'expected_features' in globals() else 0
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": "model" in globals(),
        "api_version": "1.0"
    }