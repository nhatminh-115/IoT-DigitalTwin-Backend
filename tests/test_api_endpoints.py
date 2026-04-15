from __future__ import annotations

import base64
import io


def _assert_error_schema(payload: dict) -> None:
    assert "error" in payload
    err = payload["error"]
    assert isinstance(err, dict)
    assert "code" in err
    assert "message" in err
    assert "trace_id" in err
    assert isinstance(err["trace_id"], str)
    assert len(err["trace_id"]) > 0


def test_health_schema_when_model_not_fitted(client, mock_service) -> None:
    mock_service.model_fitted = False

    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["model_fitted"] is False
    assert "status" in data
    assert "data_age_seconds" in data
    assert "bg_thread_alive" in data
    assert "tg_thread_alive" in data
    assert "cache_rows" in data
    assert "last_fetch_utc" in data


def test_health_returns_degraded_when_last_error_exists(client, mock_service) -> None:
    mock_service._last_error = "Background Fetch Error: timeout"
    mock_service._data_age_seconds = 120

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "degraded"


def test_latest_returns_503_when_cache_empty(client, mock_service) -> None:
    mock_service._cached_clean_df = None

    response = client.get("/latest")

    assert response.status_code == 503
    payload = response.json()
    _assert_error_schema(payload)
    assert payload["error"]["code"] == "DATA_SOURCE_UNAVAILABLE"


def test_predict_latest_returns_structure_when_model_fitted(client, mock_service) -> None:
    mock_service.model_fitted = True

    response = client.get("/predict/latest")

    assert response.status_code == 200
    data = response.json()
    assert "forecast_issue_time" in data
    assert "forecast_target_time" in data
    assert "channels" in data
    assert isinstance(data["channels"], dict)
    assert len(data["channels"]) > 0


def test_heatmap_temp_returns_png(client, mock_service, api_module, monkeypatch) -> None:
    api_server, _ = api_module
    mock_service._cached_clean_df = mock_service._build_sample_df()

    png_1x1 = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO7fV1QAAAAASUVORK5CYII="
    )

    def _fake_heatmap(*args, **kwargs):
        return io.BytesIO(png_1x1)

    monkeypatch.setattr(api_server.viz_engine, "heatmap", _fake_heatmap)

    response = client.get("/heatmap/temp")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/png")


def test_heatmap_invalid_metric_returns_400(client) -> None:
    response = client.get("/heatmap/invalid_metric")

    assert response.status_code == 400
    payload = response.json()
    _assert_error_schema(payload)
    assert payload["error"]["code"] == "BAD_REQUEST"


def test_error_response_contains_trace_id(client, mock_service) -> None:
    mock_service.model_fitted = False

    response = client.get("/predict/latest")

    assert response.status_code == 503
    payload = response.json()
    _assert_error_schema(payload)
    assert payload["error"]["code"] == "MODEL_NOT_READY"
