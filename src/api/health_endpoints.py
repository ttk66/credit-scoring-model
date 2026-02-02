"""
Health Check Endpoints  Credit Scoring API
  endpoints  src/api/app.py ( src/api/main.py)
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
import logging
from datetime import datetime
import psycopg2
import redis

logger = logging.getLogger(__name__)
router = APIRouter(tags=["health"])


# Status model
class HealthStatus:
    def __init__(self):
        self.status = "unhealthy"
        self.timestamp = None
        self.checks = {}


@router.get("/health")
async def health_check():
    """
    Simple health check endpoint
    Used by load balancers and orchestration systems
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "credit-scoring-api",
        "version": "1.0.0"
    }


@router.get("/ready")
async def readiness_check():
    """
    Readiness probe -     
          Redis
    """
    checks = {
        "database": False,
        "redis": False,
        "model": False
    }
    
    # Check database connection
    try:
        #    
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST", "postgresql"),
            port=os.getenv("DB_PORT", "5432"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", "postgres"),
            database=os.getenv("DB_NAME", "credit_scoring"),
            timeout=2
        )
        conn.close()
        checks["database"] = True
    except Exception as e:
        logger.warning(f"Database check failed: {e}")
        checks["database"] = False
    
    # Check Redis connection
    try:
        r = redis.from_url(os.getenv("REDIS_URL", "redis://redis:6379"))
        r.ping()
        checks["redis"] = True
    except Exception as e:
        logger.warning(f"Redis check failed: {e}")
        checks["redis"] = False
    
    # Check model availability
    try:
        #     / 
        if hasattr(app.state, 'model') and app.state.model is not None:
            checks["model"] = True
        else:
            checks["model"] = False
    except Exception as e:
        logger.warning(f"Model check failed: {e}")
        checks["model"] = False
    
    all_ready = all(checks.values())
    
    status_code = 200 if all_ready else 503
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "ready" if all_ready else "not_ready",
            "timestamp": datetime.utcnow().isoformat(),
            "checks": checks
        }
    )


@router.get("/startup")
async def startup_check():
    """
    Startup probe -    
       ,     
    """
    checks = {
        "initialization": False,
        "model_loaded": False,
        "config_loaded": False
    }
    
    try:
        # Check initialization
        if hasattr(app.state, 'initialized') and app.state.initialized:
            checks["initialization"] = True
    except Exception as e:
        logger.error(f"Initialization check failed: {e}")
    
    try:
        # Check model loading
        if hasattr(app.state, 'model') and app.state.model is not None:
            checks["model_loaded"] = True
    except Exception as e:
        logger.error(f"Model loading check failed: {e}")
    
    try:
        # Check config loading
        if hasattr(app.state, 'config') and app.state.config is not None:
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
            "checks": checks
        }
    )


@router.get("/live")
async def liveness_check():
    """
    Liveness probe -   
     endpoint  , pod  
    """
    return {
        "status": "alive",
        "timestamp": datetime.utcnow().isoformat(),
        "uptime_seconds": time.time() - app.state.start_time
    }


#    MAIN APP:
"""
from fastapi import FastAPI
import time
import os

app = FastAPI(
    title="Credit Scoring API",
    description="ML API for credit scoring predictions",
    version="1.0.0"
)

#   
app.state.start_time = time.time()
app.state.initialized = False
app.state.model = None
app.state.config = None


@app.on_event("startup")
async def startup_event():
    #    
    logger.info("Starting up...")
    
    try:
        #  
        app.state.model = load_model("/models/credit_scoring.onnx")
        
        #  
        app.state.config = load_config("/app/config/app-config.yaml")
        
        app.state.initialized = True
        logger.info("Application initialized successfully")
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        app.state.initialized = False


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down...")
    # Cleanup


#  health endpoints
app.include_router(router)

#  endpoints
@app.post("/predict")
async def predict(features: dict):
    if not app.state.initialized or app.state.model is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    #  
    ...


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""
