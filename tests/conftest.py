from __future__ import annotations

import importlib
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.iot_digital_twin.api_service import ApiServiceError


class MockInferenceAPIService:
    """Offline stand-in for InferenceAPIService used by API tests."""

    def __init__(self) -> None:
        self._cached_clean_df: pd.DataFrame | None = self._build_sample_df()
        self.model_fitted = True
        self._checkpoint_checked = True
        self._last_prediction_utc: str | None = None
        self._last_error: str | None = None
        self._last_fetch_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self._data_age_seconds: int | None = 60
        self._bg_thread_alive = True
        self._tg_thread_alive = True

    @staticmethod
    def _build_sample_df() -> pd.DataFrame:
        ts = datetime.now(timezone.utc)
        rows = [
            {
                "M1_Temp": 29.0,
                "M1_Humid": 62.0,
                "M1_CO2": 680.0,
                "M1_TVOC": 110.0,
                "M4_Temp": 30.0,
                "M4_Humid": 60.0,
                "M4_CO2": 720.0,
                "M4_TVOC": 130.0,
            },
            {
                "M1_Temp": 29.4,
                "M1_Humid": 61.0,
                "M1_CO2": 700.0,
                "M1_TVOC": 120.0,
                "M4_Temp": 30.2,
                "M4_Humid": 59.0,
                "M4_CO2": 740.0,
                "M4_TVOC": 140.0,
            },
        ]
        index = pd.DatetimeIndex([ts - timedelta(minutes=3), ts])
        return pd.DataFrame(rows, index=index)

    def health(self) -> dict[str, Any]:
        data_age_seconds = self._data_age_seconds
        status = "ok"
        if (not self._bg_thread_alive) or (
            data_age_seconds is not None and data_age_seconds > 900
        ):
            status = "unhealthy"
        elif self._last_error or data_age_seconds is None or data_age_seconds >= 300:
            status = "degraded"
        return {
            "status": status,
            "model_fitted": self.model_fitted,
            "checkpoint_path": "artifacts/lstm_checkpoint.pt",
            "checkpoint_checked": self._checkpoint_checked,
            "last_prediction_utc": self._last_prediction_utc,
            "last_error": self._last_error,
            "last_fetch_utc": self._last_fetch_utc,
            "data_age_seconds": data_age_seconds,
            "bg_thread_alive": self._bg_thread_alive,
            "tg_thread_alive": self._tg_thread_alive,
            "cache_rows": None
            if self._cached_clean_df is None
            else int(len(self._cached_clean_df)),
        }

    def get_latest_raw_data(self) -> dict[str, Any]:
        if self._cached_clean_df is None or self._cached_clean_df.empty:
            raise ApiServiceError("fetch data unavailable")
        latest = self._cached_clean_df.iloc[-1]
        return {
            "ThoiGian": self._cached_clean_df.index[-1].isoformat(),
            "M1_Temp": f"{float(latest['M1_Temp']):.2f}",
            "M1_Hum": f"{float(latest['M1_Humid']):.2f}",
            "M1_CO2": f"{float(latest['M1_CO2']):.2f}",
            "M1_TVOC": f"{float(latest['M1_TVOC']):.2f}",
        }

    def get_latest_day_raw_data(self) -> dict[str, Any]:
        return self._history_payload()

    def get_latest_hour_raw_data(self) -> dict[str, Any]:
        return self._history_payload()

    def get_latest_7_days_raw_data(self) -> dict[str, Any]:
        return self._history_payload()

    def _history_payload(self) -> dict[str, Any]:
        if self._cached_clean_df is None or self._cached_clean_df.empty:
            raise ApiServiceError("fetch data unavailable")
        return {
            "date": self._cached_clean_df.index[-1].strftime("%Y-%m-%d"),
            "count": int(len(self._cached_clean_df)),
            "data": [],
        }

    def get_latest_prediction(self, module: str | None = None) -> dict[str, Any]:
        if self._cached_clean_df is None or self._cached_clean_df.empty:
            raise ApiServiceError("fetch data unavailable")
        if not self.model_fitted:
            raise ApiServiceError("model not fitted")

        clean_df = self._cached_clean_df
        latest = clean_df.iloc[-1]
        channels: dict[str, dict[str, float]] = {}
        for column in clean_df.columns:
            if module is not None and not column.startswith(f"{module}_"):
                continue
            current = float(latest[column])
            channels[column] = {
                "current": current,
                "predicted": current + 1.0,
                "delta": 1.0,
            }

        if module is not None and not channels:
            raise ApiServiceError(f"data module {module} unavailable")

        self._last_prediction_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
        return {
            "forecast_issue_time": clean_df.index[-1].isoformat(),
            "forecast_target_time": (
                clean_df.index[-1] + timedelta(minutes=3)
            ).isoformat(),
            "module": module,
            "channels": channels,
        }

    def get_unity_payload(self, module: str) -> dict[str, Any]:
        snapshot = self.get_latest_prediction(module=module)
        return {
            "module": module,
            "forecast_issue_time": snapshot["forecast_issue_time"],
            "forecast_target_time": snapshot["forecast_target_time"],
            "features": [],
        }

    def get_unity_payload_all(self) -> dict[str, Any]:
        return {"modules": ["M1", "M4"]}

    def retrain_now(self) -> dict[str, Any]:
        return {"status": "ok"}


@pytest.fixture
def api_module(monkeypatch):
    from src.iot_digital_twin import api_service as api_service_module

    mock_service = MockInferenceAPIService()

    def _build_mock_service(_config):
        return mock_service

    monkeypatch.setattr(api_service_module, "InferenceAPIService", _build_mock_service)

    sys.modules.pop("api_server", None)
    api_server = importlib.import_module("api_server")
    return api_server, mock_service


@pytest.fixture
def mock_service(api_module) -> MockInferenceAPIService:
    return api_module[1]


@pytest.fixture
def client(api_module) -> TestClient:
    api_server, _ = api_module
    return TestClient(api_server.app)
