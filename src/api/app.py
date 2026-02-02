from fastapi import FastAPI, HTTPException, APIRouter
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
import logging
import os
import time
import threading
from collections import deque
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import redis
import joblib
import onnxruntime as ort
import pandas as pd
import numpy as np
from prometheus_client import Counter, Gauge, generate_latest, CONTENT_TYPE_LATEST
from sklearn.metrics import roc_auc_score

try:
    import psycopg2
except ImportError:
    psycopg2 = None

try:
    from src.monitoring.drift_detection import DriftDetector
except Exception:
    DriftDetector = None

MODEL_PATH = Path(os.getenv("MODEL_PATH", "/models/nn_model.onnx"))
SCALER_PATH = Path(os.getenv("SCALER_PATH", "/models/nn_scaler.joblib"))
DRIFT_CHECK_INTERVAL_SECONDS = int(os.getenv("DRIFT_CHECK_INTERVAL_SECONDS", "300"))
DRIFT_WINDOW_SIZE = int(os.getenv("DRIFT_WINDOW_SIZE", "500"))
DRIFT_MIN_SAMPLES = int(os.getenv("DRIFT_MIN_SAMPLES", "100"))
PERFORMANCE_DECAY_THRESHOLD = float(os.getenv("PERFORMANCE_DECAY_THRESHOLD", "0.05"))
DRIFT_BASELINE_PATH = Path(os.getenv("DRIFT_BASELINE_PATH", "/app/data/drift_baseline.csv"))

PREDICTIONS_TOTAL = Counter("model_predictions_total", "Total number of model predictions")
PREDICTION_ERRORS_TOTAL = Counter("model_prediction_errors_total", "Total number of prediction errors")
DATA_DRIFT_DETECTED = Gauge("data_drift_detected", "Data drift detection flag")
EVIDENTLY_FEATURE_DRIFT_SCORE = Gauge("evidently_feature_drift_score", "Share of features with drift")
EVIDENTLY_PREDICTION_DRIFT_SCORE = Gauge("evidently_prediction_drift_score", "Prediction drift score")
EVIDENTLY_TARGET_DRIFT_SCORE = Gauge("evidently_target_drift_score", "Target drift/performance decay score")
MODEL_PERFORMANCE_DECAY = Gauge("model_performance_decay", "Performance decay detection flag")
CONCEPT_DRIFT_DETECTED = Gauge("concept_drift_detected", "Concept drift detection flag")

app = FastAPI(
    title="Credit Scoring Model API",
    description="REST API for default prediction",
    version="1.0",
)

# Logger
logger = logging.getLogger(__name__)

# Initialize app state placeholders
app.state.start_time = time.time()
app.state.initialized = False
app.state.model = None
app.state.config = None
app.state.model_type = None
app.state.model_input_name = None
app.state.scaler = None
app.state.feature_window = deque(maxlen=DRIFT_WINDOW_SIZE)
app.state.prediction_window = deque(maxlen=DRIFT_WINDOW_SIZE)
app.state.labeled_window = deque(maxlen=DRIFT_WINDOW_SIZE)
app.state.prediction_index = {}
app.state.reference_df = None
app.state.drift_detector = None
app.state.baseline_auc = None
app.state.stop_drift_thread = False


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


class FeedbackPayload(BaseModel):
    request_id: str
    label: int


def build_features_api(df: pd.DataFrame) -> pd.DataFrame:
    """  build_features  API"""
    df = df.copy()

    mask = df["limit_bal"] > 0
    df["limit_age_ratio"] = np.where(mask, df["limit_bal"] / (df["age"] + 1), 0)

    bill_cols = [f"bill_amt{i}" for i in range(1, 7)]
    pay_cols = [f"pay_amt{i}" for i in range(1, 7)]

    df["avg_bill_amt"] = df[bill_cols].mean(axis=1)
    df["avg_pay_amt"] = df[pay_cols].mean(axis=1)

    denom = df["avg_bill_amt"]
    valid_mask = denom > 0

    df["bill_pay_ratio"] = np.nan
    if valid_mask.any():
        df.loc[valid_mask, "bill_pay_ratio"] = (
            df.loc[valid_mask, "avg_pay_amt"] / denom[valid_mask]
        )

    df["bill_pay_ratio"] = pd.to_numeric(df["bill_pay_ratio"], errors="coerce")
    df["bill_pay_ratio"] = df["bill_pay_ratio"].fillna(0)
    df["bill_pay_ratio"] = df["bill_pay_ratio"].clip(lower=0, upper=5)

    #  pay_1   
    if "pay_1" not in df.columns:
        df["pay_1"] = 0

    pay_status_cols = [f"pay_{i}" for i in range(0, 7)]
    df["num_late_payments"] = (df[pay_status_cols] > 0).sum(axis=1).astype(int)
    df["max_delay"] = df[pay_status_cols].max(axis=1)

    df_positive = df[pay_status_cols].where(df[pay_status_cols] > 0)
    df["avg_delay"] = df_positive.mean(axis=1)
    df["avg_delay"] = df["avg_delay"].fillna(0)

    #  
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

    #  
    bins = [0, 30, 40, 50, 60, 100]
    labels = [0, 1, 2, 3, 4]

    df["age_bin"] = pd.cut(df["age"], bins=bins, labels=labels, include_lowest=True)
    df["age_bin"] = df["age_bin"].cat.add_categories([-1]).fillna(-1)

    #  
    for c in ["sex", "education", "marriage", "age_bin"]:
        df[c] = df[c].astype("category")

    return df


def _calculate_auc_safe(rows):
    if len(rows) < DRIFT_MIN_SAMPLES:
        return None
    y_true = np.array([r["label"] for r in rows], dtype=np.int32)
    y_pred = np.array([r["prediction"] for r in rows], dtype=np.float32)
    # AUC is undefined if only one class is present.
    if len(np.unique(y_true)) < 2:
        return None
    try:
        return float(roc_auc_score(y_true, y_pred))
    except Exception:
        return None


def _load_baseline(expected_features_list):
    if not DRIFT_BASELINE_PATH.exists():
        return pd.DataFrame(columns=expected_features_list)
    try:
        baseline = pd.read_csv(DRIFT_BASELINE_PATH)
        missing = [c for c in expected_features_list if c not in baseline.columns]
        for col in missing:
            baseline[col] = 0.0
        return baseline[expected_features_list]
    except Exception as exc:
        logger.warning("Could not load drift baseline: %s", exc)
        return pd.DataFrame(columns=expected_features_list)


def _save_baseline(df):
    try:
        DRIFT_BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(DRIFT_BASELINE_PATH, index=False)
    except Exception as exc:
        logger.warning("Could not save drift baseline: %s", exc)


def _drift_monitor_loop():
    while not app.state.stop_drift_thread:
        try:
            feature_rows = list(app.state.feature_window)
            pred_rows = np.array(list(app.state.prediction_window), dtype=np.float32)

            if (
                app.state.drift_detector is not None
                and len(feature_rows) >= DRIFT_MIN_SAMPLES
            ):
                current_df = pd.DataFrame(feature_rows)
                if app.state.reference_df is None or app.state.reference_df.empty:
                    app.state.reference_df = current_df.copy()
                    _save_baseline(app.state.reference_df)
                    app.state.drift_detector = DriftDetector(
                        reference_data=app.state.reference_df,
                        feature_names=current_df.columns.tolist(),
                    )
                drift_result = app.state.drift_detector.detect_data_drift(current_df)
                drift_flag = 1.0 if drift_result.get("drift_detected") else 0.0
                DATA_DRIFT_DETECTED.set(drift_flag)
                EVIDENTLY_FEATURE_DRIFT_SCORE.set(
                    float(drift_result.get("drift_percentage", 0.0))
                )

            if app.state.reference_df is not None and len(pred_rows) >= DRIFT_MIN_SAMPLES:
                ref_pred = app.state.reference_df["prediction"].to_numpy(dtype=np.float32)
                if len(ref_pred) >= DRIFT_MIN_SAMPLES:
                    from scipy.stats import ks_2samp

                    ks_stat, _ = ks_2samp(ref_pred, pred_rows)
                    EVIDENTLY_PREDICTION_DRIFT_SCORE.set(float(ks_stat))
                    CONCEPT_DRIFT_DETECTED.set(1.0 if ks_stat > 0.2 else 0.0)

            labeled_rows = list(app.state.labeled_window)
            auc_now = _calculate_auc_safe(labeled_rows)
            if auc_now is not None:
                if app.state.baseline_auc is None:
                    app.state.baseline_auc = auc_now
                decay = app.state.baseline_auc - auc_now
                EVIDENTLY_TARGET_DRIFT_SCORE.set(float(decay))
                MODEL_PERFORMANCE_DECAY.set(1.0 if decay > PERFORMANCE_DECAY_THRESHOLD else 0.0)
                if decay > PERFORMANCE_DECAY_THRESHOLD:
                    CONCEPT_DRIFT_DETECTED.set(1.0)

        except Exception as exc:
            logger.warning("Drift monitor iteration failed: %s", exc)

        time.sleep(DRIFT_CHECK_INTERVAL_SECONDS)


@app.on_event("startup")
def load_model():
    global model, expected_features
    # initialize app state placeholders (startup time, readiness)
    app.state.start_time = time.time()
    app.state.initialized = False
    app.state.model = None
    app.state.config = None

    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model not found: {MODEL_PATH}")

    expected_features = [
        "limit_bal",
        "sex",
        "education",
        "marriage",
        "age",
        "pay_0",
        "pay_2",
        "pay_3",
        "pay_4",
        "pay_5",
        "pay_6",
        "bill_amt1",
        "bill_amt2",
        "bill_amt3",
        "bill_amt4",
        "bill_amt5",
        "bill_amt6",
        "pay_amt1",
        "pay_amt2",
        "pay_amt3",
        "pay_amt4",
        "pay_amt5",
        "pay_amt6",
        "limit_age_ratio",
        "avg_bill_amt",
        "avg_pay_amt",
        "bill_pay_ratio",
        "num_late_payments",
        "max_delay",
        "avg_delay",
        "bill_trend",
        "age_bin",
    ]

    if MODEL_PATH.suffix == ".onnx":
        session = ort.InferenceSession(
            str(MODEL_PATH), providers=["CPUExecutionProvider"]
        )
        app.state.model = session
        app.state.model_type = "onnx"
        app.state.model_input_name = session.get_inputs()[0].name
        print(f"ONNX model loaded: {MODEL_PATH}")
        print(f"ONNX input: {app.state.model_input_name}")

        if SCALER_PATH.exists():
            app.state.scaler = joblib.load(SCALER_PATH)
            print(f"Scaler loaded: {SCALER_PATH}")
        else:
            print(f"Scaler not found, using unscaled features: {SCALER_PATH}")
    else:
        model = joblib.load(MODEL_PATH)
        app.state.model = model
        app.state.model_type = "sklearn"
        print(f"Sklearn model loaded: {MODEL_PATH}")

        if hasattr(model, "feature_names_in_"):
            expected_features = model.feature_names_in_.tolist()
            print(f"Model expects {len(expected_features)} features")

    if DriftDetector is None:
        logger.warning("Evidently drift detector is unavailable; drift metrics will be partial.")
    else:
        app.state.reference_df = _load_baseline(expected_features)
        try:
            app.state.drift_detector = DriftDetector(
                reference_data=app.state.reference_df, feature_names=expected_features
            )
        except Exception as exc:
            logger.warning("Could not initialize DriftDetector: %s", exc)
            app.state.drift_detector = None

    app.state.stop_drift_thread = False
    threading.Thread(target=_drift_monitor_loop, daemon=True).start()
    # mark initialized after successful load
    app.state.initialized = True


@app.on_event("shutdown")
def shutdown_event():
    app.state.stop_drift_thread = True


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

        if app.state.model_type == "onnx":
            input_data = processed_df.astype(np.float32).values
            if app.state.scaler is not None:
                input_data = app.state.scaler.transform(input_data).astype(np.float32)
            output = app.state.model.run(
                None, {app.state.model_input_name: input_data}
            )[0]
            proba = float(np.asarray(output).reshape(-1)[0])
            proba = max(0.0, min(1.0, proba))
        else:
            proba = model.predict_proba(processed_df)[0, 1]
        pred = int(proba >= 0.5)
        request_id = str(uuid4())
        app.state.feature_window.append(processed_df.iloc[0].to_dict())
        app.state.prediction_window.append(float(proba))
        app.state.prediction_index[request_id] = {
            "prediction": float(proba),
            "ts": time.time(),
        }
        # Keep index bounded to avoid unbounded memory growth.
        if len(app.state.prediction_index) > DRIFT_WINDOW_SIZE * 4:
            oldest_key = next(iter(app.state.prediction_index))
            app.state.prediction_index.pop(oldest_key, None)
        PREDICTIONS_TOTAL.inc()

        return {
            "prediction": pred,
            "probability": float(proba),
            "features_processed": processed_df.shape[1],
            "request_id": request_id,
        }

    except Exception as e:
        PREDICTION_ERRORS_TOTAL.inc()
        import traceback

        error_details = f"{str(e)}\n\n{traceback.format_exc()}"
        raise HTTPException(status_code=400, detail=error_details)


@app.get("/model_info")
def get_model_info():
    """  """
    return {
        "model_type": getattr(app.state, "model_type", "unknown"),
        "pipeline_steps": (
            list(app.state.model.named_steps.keys())
            if getattr(app.state, "model_type", "") == "sklearn"
            and hasattr(app.state.model, "named_steps")
            else []
        ),
        "expected_features": (
            expected_features if "expected_features" in globals() else []
        ),
        "expected_features_count": (
            len(expected_features) if "expected_features" in globals() else 0
        ),
    }


@app.post("/feedback")
def post_feedback(payload: FeedbackPayload):
    entry = app.state.prediction_index.get(payload.request_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Unknown request_id")

    app.state.labeled_window.append(
        {"prediction": float(entry["prediction"]), "label": int(payload.label)}
    )
    return {"status": "accepted", "request_id": payload.request_id}


@app.post("/drift/baseline/refresh")
def refresh_drift_baseline():
    if len(app.state.feature_window) < DRIFT_MIN_SAMPLES:
        raise HTTPException(status_code=400, detail="Not enough samples to refresh baseline")
    baseline_df = pd.DataFrame(list(app.state.feature_window))
    app.state.reference_df = baseline_df
    if app.state.drift_detector is not None:
        app.state.drift_detector.reference_data = baseline_df
    _save_baseline(baseline_df)
    return {"status": "ok", "rows": len(baseline_df), "path": str(DRIFT_BASELINE_PATH)}


@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# Health endpoints (liveness/readiness/startup)
router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "credit-scoring-api",
        "version": "1.0.0",
    }


@router.get("/ready")
async def readiness_check():
    checks = {"database": False, "redis": False, "model": False}

    # Check database (optional in minimal deployment)
    if psycopg2 is not None:
        try:
            conn = psycopg2.connect(
                host=os.getenv("DB_HOST", "postgresql"),
                port=os.getenv("DB_PORT", "5432"),
                user=os.getenv("DB_USER", "postgres"),
                password=os.getenv("DB_PASSWORD", "postgres"),
                database=os.getenv("DB_NAME", "credit_scoring"),
                connect_timeout=2,
            )
            conn.close()
            checks["database"] = True
        except Exception as e:
            logger.warning(f"Database check failed: {e}")

    # Check Redis (optional in minimal deployment)
    try:
        r = redis.from_url(os.getenv("REDIS_URL", "redis://redis:6379"))
        r.ping()
        checks["redis"] = True
    except Exception as e:
        logger.warning(f"Redis check failed: {e}")

    # Check model
    try:
        if hasattr(app.state, "model") and app.state.model is not None:
            checks["model"] = True
    except Exception as e:
        logger.warning(f"Model check failed: {e}")

    strict_dependencies = os.getenv("STRICT_DEPENDENCY_READINESS", "false").lower() == "true"
    all_ready = all(checks.values()) if strict_dependencies else checks["model"]
    status_code = 200 if all_ready else 503
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "ready" if all_ready else "not_ready",
            "timestamp": datetime.utcnow().isoformat(),
            "checks": checks,
        },
    )


@router.get("/startup")
async def startup_check():
    checks = {"initialization": False, "model_loaded": False, "config_loaded": False}

    try:
        if hasattr(app.state, "initialized") and app.state.initialized:
            checks["initialization"] = True
    except Exception as e:
        logger.error(f"Initialization check failed: {e}")

    try:
        if hasattr(app.state, "model") and app.state.model is not None:
            checks["model_loaded"] = True
    except Exception as e:
        logger.error(f"Model loading check failed: {e}")

    try:
        if hasattr(app.state, "config") and app.state.config is not None:
            checks["config_loaded"] = True
    except Exception as e:
        logger.error(f"Config loading check failed: {e}")

    all_started = all(checks.values())
    status_code = 200 if all_started else 503
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "started" if all_started else "starting",
            "timestamp": datetime.utcnow().isoformat(),
            "checks": checks,
        },
    )


@router.get("/live")
async def liveness_check():
    return {
        "status": "alive",
        "timestamp": datetime.utcnow().isoformat(),
        "uptime_seconds": time.time() - getattr(app.state, "start_time", app.state.start_time),
    }


# Register health router
app.include_router(router)
