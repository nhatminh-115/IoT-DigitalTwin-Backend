"""FastAPI server exposing IoT Digital Twin inference endpoints."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from src.iot_digital_twin.api_service import ApiServiceConfig, ApiServiceError, InferenceAPIService
from src.iot_digital_twin import viz_engine


CSV_URL = (
    "https://docs.google.com/spreadsheets/d/e/2PACX-1vSnBhE8u7fdXKEooOlqgGYtSZLeUxrQUu9e_q6MnMrGbakxXESMYVf0utORhEG3pEqWffGhX6J-V2cC/pub?output=csv"
)
CHECKPOINT_PATH = Path("artifacts") / "lstm_checkpoint.pt"

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


@app.get("/health")
def health() -> dict:
    """Returns service liveness and model readiness details."""
    return service.health()


@app.get("/latest")
def latest_data() -> dict:
    """Returns the latest raw sensor data row as a flat JSON."""
    try:
        return service.get_latest_raw_data()
    except ApiServiceError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/latest_day")
def latest_day_data() -> dict:
    """Returns historical raw sensor data (up to 24 hours)."""
    try:
        return service.get_latest_day_raw_data()
    except ApiServiceError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/latest_hour")
def latest_hour_data() -> dict:
    """Returns historical raw sensor data (up to 1 hour)."""
    try:
        return service.get_latest_hour_raw_data()
    except ApiServiceError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/latest_7_days")
def latest_7_days_data() -> dict:
    """Returns historical raw sensor data (up to 7 days)."""
    try:
        return service.get_latest_7_days_raw_data()
    except ApiServiceError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/predict/latest")
def predict_latest(
    module: str | None = Query(default=None, description="Module filter, e.g. M1 or M9."),
) -> dict:
    """Returns the latest t+1 prediction payload, optionally filtered by module."""
    try:
        return service.get_latest_prediction(module=module)
    except ApiServiceError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/unity/predict")
def unity_predict(
    module: str = Query(default="M1", description="Module code for Unity payload."),
) -> dict:
    """Returns compact feature-level prediction payload for Unity clients."""
    try:
        return service.get_unity_payload(module=module)
    except ApiServiceError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/unity/predict/all")
def unity_predict_all() -> dict:
    """Returns grouped prediction payload for all modules in one response."""
    try:
        return service.get_unity_payload_all()
    except ApiServiceError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/retrain")
def retrain() -> dict:
    """Forces immediate retraining from newest data and updates checkpoint."""
    try:
        return service.retrain_now()
    except ApiServiceError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


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
        raise HTTPException(status_code=500, detail=f"Error generating heatmap: {exc}")
