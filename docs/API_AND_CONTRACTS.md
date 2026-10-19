# HTTP API and data contracts

**Verified against:** [`api_server.py`](../api_server.py), [`api_service.py`](../src/iot_digital_twin/api_service.py), [`data_fetcher.py`](../src/iot_digital_twin/data_fetcher.py), [`mqtt_worker.py`](../mqtt_worker.py), and the checked-in [Unity client](../unity_integration/UnityInferenceClient.cs). These examples illustrate **schema**, not captured live responses.

## URLs and API routing

| Execution | Base URL for endpoints below | Dashboard |
|---|---|---|
| Local direct Uvicorn | `http://localhost:8000` | Separate Streamlit process; documented local port `8503` |
| Docker / hosted Nginx | `http://<host>:7860/api` | `http://<host>:7860/` |
| Internal worker process | `http://127.0.0.1:8001` | Not proxied by Nginx |

`nginx.conf` removes the `/api/` prefix via `proxy_pass http://127.0.0.1:8000/`. For example, public `GET /api/health` reaches FastAPI `GET /health`. The Unity C# client's `apiBaseUrl` must be configured accordingly; the default client targets direct `http://127.0.0.1:8000`.

## Route inventory (current code, not README aliases)

| Method | Route | Returned data / action | Source |
|---|---|---|---|
| GET | `/health` | Runtime health, model status, fetch age, cache rows, thread status; in `api` mode the worker snapshot drives health | `service.health()` |
| GET | `/latest` | Latest cleaned sensor observation as a flat legacy Unity dictionary | `get_latest_raw_data()` |
| GET | `/latest_hour` | Up to 20 resampled historical rows (~1 hour) in `{date,count,data[]}` | `get_latest_hour_raw_data()` |
| GET | `/latest_day` | Up to 480 rows (~24 hours), same wrapper | `get_latest_day_raw_data()` |
| GET | `/latest_7_days` | Up to 3360 rows (~7 days), same wrapper | `get_latest_7_days_raw_data()` |
| GET | `/predict/latest` | t+1 per-channel forecast; optional `?module=M1` filter | `get_latest_prediction()` |
| GET | `/unity/predict` | Compact Unity-friendly features for one module; default `?module=M1` | `get_unity_payload()` |
| GET | `/unity/predict/all` | All available modules as `modules[]` | `get_unity_payload_all()` |
| POST | `/retrain` | Immediately fit/update LSTM from current cached data and overwrite checkpoint | `retrain_now()` |
| GET | `/heatmap/{metric}` | PNG image; metric must be one of `temp`, `humid`, `co2`, `tvoc` | `viz_engine.heatmap()` |
| POST | `/command` | Store the latest control command in this **API process's memory** | `latest_command` global |
| GET | `/command` | Return last command from this **API process's memory** | `latest_command` global |
| GET | `/control` | Serve HTML remote-control page from repository | `remote_control_dashboard.html` |

**Not actual paths:** some older README examples mention `/predict`, `/latest/hour` or `/latest` returning a nested `modules` object. The handlers in `api_server.py` instead use `/predict/latest`, `/latest_hour` and a **flat** `/latest` object. Use the table above as the code-grounded contract.

## Source CSV → canonical DataFrame

Input is an HTTP CSV export from a Google Sheet. Its exact columns can vary; `DataFetcher` recognizes time-like headers (for example `timestamp`, `datetime`, `date`, `time`), device headers (`device id`, `device`, `node id`, `node`), and common metric aliases:

| Accepted input names | Canonical wide-frame column pattern | Notes |
|---|---|---|
| `temp`, `temperature`, `nhiệt độ` | `M{n}_Temp` | Numeric |
| `humidity`, `hum`, `độ ẩm` | `M{n}_Humid` | Internal spelling `Humid` |
| `eco2`, `co2` | `M{n}_CO2` | `eCO2` input accepted; actual measurement semantics must be verified independently |
| `tvoc` | `M{n}_TVOC` | Numeric |
| `pm 1.0`/`pm1` etc. | `M{n}_PM1` | Optional if present |
| `pm 2.5`/`pm25` etc. | `M{n}_PM25` | Optional if present |
| `pm 10`/`pm10` | `M{n}_PM10` | Optional if present |

The normalizer maps digit-bearing device IDs such as `esp09` → `M9`, keeps device numbers 1–16, pivots rows to a wide time index, resamples at `3min`, and performs initial interpolation. The model output schema follows the columns present after data quality filtering; **do not assume only four features are always present**.

Illustrative input:

```csv
Timestamp,Device ID,Temp,Humidity,eCO2,TVOC
2026-10-19 08:00:00,esp09,28.2,68,450,20
2026-10-19 08:00:00,esp01,27.4,65,420,17
```

Illustrative normalized columns: `M9_Temp`, `M9_Humid`, `M9_CO2`, `M9_TVOC`, `M1_Temp`, `M1_Humid`, `M1_CO2`, `M1_TVOC`.

## Latest and historical reading contract

`GET /latest` returns a **flat dictionary** whose keys are `Timestamp` and sensor channels. Numeric channel values are formatted to **two decimal places as strings** and `_Humid` is renamed to `_Hum` for legacy Unity compatibility.

```json
{
  "Timestamp": "2026-10-19 08:00:00",
  "M1_Temp": "27.40",
  "M1_Hum": "65.00",
  "M1_CO2": "420.00",
  "M1_TVOC": "17.00"
}
```

Historical endpoints (`/latest_hour`, `/latest_day`, `/latest_7_days`) return:

```json
{
  "date": "2026-10-19",
  "count": 1,
  "data": [
    {
      "M1_Temp": "27.40",
      "M1_Hum": "65.00",
      "M1_CO2": "420.00",
      "M1_TVOC": "17.00",
      "created": "08:00:00"
    }
  ]
}
```

These history rows carry **time-of-day only** in `created` plus one outer `date` (the latest frame date), so that legacy shape is potentially ambiguous across multi-day histories. Clients requiring fully qualified per-row timestamps should not infer them from `created` without an independent date rule.

## Forecast contract

`GET /predict/latest?module=M1` filters canonical channel names to the chosen prefix. No filter returns all channels. Values are JSON numeric fields; unlike `/latest`, forecasts are **not converted to 2-decimal strings**.

```json
{
  "forecast_issue_time": "2026-10-19T08:00:00",
  "forecast_target_time": "2026-10-19T08:03:00",
  "module": "M1",
  "channels": {
    "M1_Temp": {
      "current": 27.4,
      "predicted": 27.6,
      "delta": 0.2
    },
    "M1_Humid": {
      "current": 65.0,
      "predicted": 64.8,
      "delta": -0.2
    }
  }
}
```

Times are derived from the cleaned DataFrame's timestamps; the code does not unconditionally attach UTC offsets to these timestamp strings. Consumers should **not assume the timestamp is UTC merely because it is ISO formatted**. The next step is the median observed interval (typically 3 minutes).

### Unity prediction contract

`GET /unity/predict?module=M1` wraps the same inference in a feature-centric form compatible with C# `JsonUtility`:

```json
{
  "module": "M1",
  "forecast_issue_time": "2026-10-19T08:00:00",
  "forecast_target_time": "2026-10-19T08:03:00",
  "current": {"Temp": 27.4, "Humid": 65.0},
  "predicted": {"Temp": 27.6, "Humid": 64.8},
  "delta": {"Temp": 0.2, "Humid": -0.2},
  "features": [
    {"name": "Temp", "current": 27.4, "predicted": 27.6, "delta": 0.2},
    {"name": "Humid", "current": 65.0, "predicted": 64.8, "delta": -0.2}
  ]
}
```

The included [UnityInferenceClient.cs](../unity_integration/UnityInferenceClient.cs) deserializes `module`, issue/target times and `features[]`; it does **not** require direct deserialization of the three dictionaries. `UnityInferenceUIBinder.cs` looks up `Temp`/`Humid`/`CO2`/`TVOC` and renders current, prediction and delta.

`GET /unity/predict/all` returns an outer object with `forecast_issue_time`, `forecast_target_time` and `modules[]`, where each item has the one-module structure shown above.

## Remote command contract

`POST /command` accepts JSON and extracts `metric`; if omitted it defaults to `home`. **It does not validate a strict command enumeration**. A successful response is an acknowledgement of storing the value in process memory, not of execution by Unity or another consumer:

```json
{
  "status": "success",
  "command": {
    "metric": "temp",
    "timestamp": 1792371600.0
  }
}
```

`GET /command` returns just the stored `{"metric":"...", "timestamp":...}` object. Default at import is `{"metric":"home", "timestamp":0.0}`. With two Uvicorn API workers, POST and GET may hit different memory copies. See [limitations](OPERATIONS_AND_CAVEATS.md#known-implementation-caveats).

## MQTT → Supabase record contract

From `esp/esp09` and the JSON payload `{"t1":28.2,"h1":68,"co2":450,"tvoc":20}`, the worker constructs:

```json
{
  "ts": "2026-10-19T01:00:00+00:00",
  "node_id": "M9",
  "temp": 28.2,
  "humidity": 68.0,
  "co2": 450,
  "tvoc": 20
}
```

`ts` is produced by `datetime.now(timezone.utc)` in the MQTT worker (receipt time), **not taken from a sensor payload field**. `temp` and `humidity` are floating-point values; `co2` and `tvoc` are integer conversions; malformed numbers map to `null`. Unknown device topic suffixes and invalid JSON are skipped. Records are upserted to `env_readings` with conflict key `ts,node_id`. This describes the **writer's expected table contract**, not an independently verified Supabase database migration/schema.

## Errors, readiness and operational behavior

FastAPI handlers register a structured error envelope:

```json
{
  "error": {
    "code": "DATA_SOURCE_UNAVAILABLE",
    "message": "System is warming up or fetching data. Please wait.",
    "trace_id": "illustrative-trace-id"
  }
}
```

| Typical condition | HTTP status | Error code |
|---|---|---|
| No cached frame / fetch failure | 503 | `DATA_SOURCE_UNAVAILABLE` |
| Model unavailable / not fitted | 503 | `MODEL_NOT_READY` |
| Validation/data-quality failure | 422 | `BAD_DATA` |
| Explicit bad HTTP request (e.g. invalid heatmap metric) | 400 | `BAD_REQUEST` |
| Explicit missing route/resource | 404 | `NOT_FOUND` |
| Unexpected error or uncategorized service error | 500 | `INTERNAL_ERROR` |

`trace_id` is generated for each error. Error classification relies partly on inspecting `ApiServiceError` message text, so a new/uncategorized exception may map to `INTERNAL_ERROR` rather than a semantically ideal code.

`GET /health` returns a status (depending on mode, local or worker-state-derived) plus fields such as `model_fitted`, `checkpoint_path`, `checkpoint_checked`, `last_prediction_utc`, `last_fetch_utc`, `last_error`, `data_age_seconds`, `bg_thread_alive`, `tg_thread_alive` and `cache_rows`. In `api` mode, worker state older than 300 seconds is degraded and older than 900 seconds unhealthy. **A healthy worker-state response is not proof that the particular API process handling a forecast has a fresh local cache.**

## Public surface and security note

No explicit authentication dependency or middleware is registered for these routes; CORS origins and methods are permissive in the checked-in API. Treat `POST /retrain` and `POST /command` as sensitive **even though they currently have no endpoint authorization gate in source**. Deploy behind appropriate access controls instead of treating this document as a security guarantee.

For the complete runtime picture, see [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md) and [DATA_FLOWS.md](DATA_FLOWS.md).
