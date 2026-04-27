"""FastAPI server exposing IoT Digital Twin inference endpoints."""

from __future__ import annotations

import logging
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from src.iot_digital_twin.api_service import ApiServiceConfig, ApiServiceError, InferenceAPIService
from src.iot_digital_twin import viz_engine


CSV_URL = (
    "https://docs.google.com/spreadsheets/d/e/2PACX-1vSnBhE8u7fdXKEooOlqgGYtSZLeUxrQUu9e_q6MnMrGbakxXESMYVf0utORhEG3pEqWffGhX6J-V2cC/pub?output=csv"
)
CHECKPOINT_PATH = Path("artifacts") / "lstm_checkpoint.pt"

logger = logging.getLogger(__name__)

app = FastAPI(
    title="IoT Digital Twin Inference API",
    version="1.0.0",
    description="REST API for real-time multivariate forecasting used by Unity and external clients.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

service = InferenceAPIService(
    ApiServiceConfig(
        csv_url=CSV_URL,
        checkpoint_path=CHECKPOINT_PATH,
    )
)


def _error_response(status_code: int, code: str, message: str, trace_id: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": code,
                "message": message,
                "trace_id": trace_id,
            }
        },
    )


def _map_api_service_error(exc: ApiServiceError) -> tuple[str, int]:
    message = str(exc).lower()
    if "fetch" in message:
        return "DATA_SOURCE_UNAVAILABLE", status.HTTP_503_SERVICE_UNAVAILABLE
    if "model" in message or "not fitted" in message:
        return "MODEL_NOT_READY", status.HTTP_503_SERVICE_UNAVAILABLE
    if "data" in message or "quality" in message:
        return "BAD_DATA", status.HTTP_422_UNPROCESSABLE_ENTITY
    return "INTERNAL_ERROR", status.HTTP_500_INTERNAL_SERVER_ERROR


def _map_http_exception_code(exc: HTTPException) -> str:
    if exc.status_code == status.HTTP_400_BAD_REQUEST:
        return "BAD_REQUEST"
    if exc.status_code == status.HTTP_404_NOT_FOUND:
        return "NOT_FOUND"
    if exc.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY:
        return "BAD_DATA"
    if exc.status_code == status.HTTP_503_SERVICE_UNAVAILABLE:
        return "DATA_SOURCE_UNAVAILABLE"
    if exc.status_code >= status.HTTP_500_INTERNAL_SERVER_ERROR:
        return "INTERNAL_ERROR"
    return "HTTP_ERROR"


@app.exception_handler(ApiServiceError)
async def handle_api_service_error(request: Request, exc: ApiServiceError) -> JSONResponse:
    trace_id = uuid4().hex
    code, status_code = _map_api_service_error(exc)
    logger.warning(
        "ApiServiceError trace_id=%s path=%s code=%s detail=%s",
        trace_id,
        request.url.path,
        code,
        str(exc),
    )
    return _error_response(status_code=status_code, code=code, message=str(exc), trace_id=trace_id)


@app.exception_handler(RequestValidationError)
async def handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    trace_id = uuid4().hex
    logger.warning(
        "RequestValidationError trace_id=%s path=%s detail=%s",
        trace_id,
        request.url.path,
        exc.errors(),
    )
    return _error_response(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        code="BAD_DATA",
        message="Request validation failed.",
        trace_id=trace_id,
    )


@app.exception_handler(HTTPException)
async def handle_http_exception(request: Request, exc: HTTPException) -> JSONResponse:
    trace_id = uuid4().hex
    message = exc.detail if isinstance(exc.detail, str) else "HTTP error."
    code = _map_http_exception_code(exc)
    logger.warning(
        "HTTPException trace_id=%s path=%s status=%s code=%s detail=%s",
        trace_id,
        request.url.path,
        exc.status_code,
        code,
        message,
    )
    return _error_response(status_code=exc.status_code, code=code, message=message, trace_id=trace_id)


@app.exception_handler(Exception)
async def handle_unexpected_exception(request: Request, exc: Exception) -> JSONResponse:
    trace_id = uuid4().hex
    logger.exception("Unhandled exception trace_id=%s path=%s", trace_id, request.url.path)
    return _error_response(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        code="INTERNAL_ERROR",
        message="Internal server error.",
        trace_id=trace_id,
    )


@app.get("/health")
def health() -> dict:
    """Returns service liveness and model readiness details."""
    return service.health()


@app.get("/latest")
def latest_data() -> dict:
    """Returns the latest raw sensor data row as a flat JSON."""
    return service.get_latest_raw_data()


@app.get("/latest_day")
def latest_day_data() -> dict:
    """Returns historical raw sensor data (up to 24 hours)."""
    return service.get_latest_day_raw_data()


@app.get("/latest_hour")
def latest_hour_data() -> dict:
    """Returns historical raw sensor data (up to 1 hour)."""
    return service.get_latest_hour_raw_data()


@app.get("/latest_7_days")
def latest_7_days_data() -> dict:
    """Returns historical raw sensor data (up to 7 days)."""
    return service.get_latest_7_days_raw_data()


@app.get("/predict/latest")
def predict_latest(
    module: str | None = Query(default=None, description="Module filter, e.g. M1 or M9."),
) -> dict:
    """Returns the latest t+1 prediction payload, optionally filtered by module."""
    return service.get_latest_prediction(module=module)


@app.get("/unity/predict")
def unity_predict(
    module: str = Query(default="M1", description="Module code for Unity payload."),
) -> dict:
    """Returns compact feature-level prediction payload for Unity clients."""
    return service.get_unity_payload(module=module)


@app.get("/unity/predict/all")
def unity_predict_all() -> dict:
    """Returns grouped prediction payload for all modules in one response."""
    return service.get_unity_payload_all()


@app.post("/retrain")
def retrain() -> dict:
    """Forces immediate retraining from newest data and updates checkpoint."""
    return service.retrain_now()


@app.get("/heatmap/{metric}")
def get_heatmap(metric: str) -> StreamingResponse:
    """Returns a heatmap image for the specified metric (temp, humid, co2, tvoc)."""
    if metric not in ["temp", "humid", "co2", "tvoc"]:
        raise HTTPException(status_code=400, detail="Invalid metric.")
    
    clean_df = service._cached_clean_df
    if clean_df is None or clean_df.empty:
        raise HTTPException(status_code=503, detail="Data not available yet. Please try again.")

    try:
        coords_path = Path("node_coords_v1.json")
        image_path = Path("campus_3d_1.png")
        
        buf = viz_engine.heatmap(
            clean_df, 
            metric=metric,
            coords_path=coords_path if coords_path.exists() else None,
            image_path=image_path if image_path.exists() else None
        )
        return StreamingResponse(buf, media_type="image/png")
    except Exception as exc:
        logger.exception("Error generating heatmap metric=%s", metric)
        raise HTTPException(status_code=500, detail="Error generating heatmap.") from exc
