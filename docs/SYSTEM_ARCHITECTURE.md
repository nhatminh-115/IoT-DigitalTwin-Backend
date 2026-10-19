# System architecture (as implemented)

**Reviewed:** 2026-10-19. **Scope:** this backend repository and the directly referenced Unity scripts. **Source of truth:** linked source code and deployment files, not a conceptual architecture drawing.

## System context

This is a Python-based environmental digital-twin backend with two independent ways of receiving telemetry. The **model-serving pipeline consumes a published Google Sheets CSV**, while the **MQTT ingestion worker writes independently to Supabase**. It provides a Streamlit dashboard, FastAPI REST service for Unity/external clients, Telegram reporting/alerts, an optional Groq-assisted question-answering path, and optional daily heatmap videos and weather cross-checks.

~~~mermaid
flowchart LR
  HA["Home Assistant / upstream automation<br/>(documented; code outside repo)"] --> GS["Google Sheets CSV export"]
  GS --> API["FastAPI + InferenceAPIService"]
  GS --> UI["Streamlit dashboard"]
  API --> Unity["Unity clients / REST consumers"]
  API --> TG["Telegram Bot API / chat users"]
  API --> Logs["Supabase: alert_logs, bot_logs,<br/>system_heartbeat (optional)"]
  API --> Weather["Open-Meteo (optional checks)"]
  API --> Groq["Groq API (optional /ask)"]
  API --> HFAssets["HF dataset: generated videos (optional)"]

  ESP["ESP devices<br/>(outside repo)"] --> Hive["HiveMQ Cloud"]
  Hive --> MQTT["mqtt_worker.py"]
  MQTT --> DB["Supabase: env_readings"]

  style GS fill:#eef
  style DB fill:#fee
~~~

**Critical non-connection:** the MQTT/Supabase `env_readings` path does **not** feed `InferenceAPIService`, the LSTM, or the Streamlit dashboard in this implementation. The original README documents Home Assistant publishing on an approximately 3-minute cadence; that upstream publisher's code is not present here. Diagram edges describe code paths or explicitly attributed external integrations, not monitored production uptime.

## Logical components

| Layer | Component | Responsibility and state | Source |
|---|---|---|---|
| Upstream source | Home Assistant → Google Sheets | External publisher of sensor records; upstream implementation not in this repo | [README](../README.md) |
| Ingestion adapter | `DataFetcher` | HTTP GET CSV (timeout/retries); timestamp/device/feature normalization; wide pivot; 3-minute resampling/interpolation | [data_fetcher.py](../src/iot_digital_twin/data_fetcher.py) |
| Data validation | `DataQualityGate` | Minimum rows, rolling-variance flatlines, per-column z-score filtering, interpolate/fill; returns a quality report | [data_quality_gate.py](../src/iot_digital_twin/data_quality_gate.py) |
| Orchestration | `InferenceAPIService` | 30-second background fetch, process-local clean-data cache, model preparation, alert/bot/auxiliary jobs, API-facing methods | [api_service.py](../src/iot_digital_twin/api_service.py) |
| Forecasting | `DeepTimeSeriesPredictor` | PyTorch multivariate LSTM, differenced/standardized feature windows, t+1 inference, checkpoint save/load | [predictor.py](../src/iot_digital_twin/predictor.py) |
| Unsupervised detection | `AnomalyDetector` | Loads saved autoencoder + metadata if present, scores reconstruction error of recent window; optional | [anomaly_detector.py](../src/iot_digital_twin/anomaly_detector.py) |
| REST entry point | `api_server.py` | Endpoints for health, latest/history, predictions, Unity, heatmap, retrain, control commands | [api_server.py](../api_server.py) |
| Web UI | `StreamlitDashboard` | **Own** fetch/quality/inference/evaluation flow, 180-second default UI refresh, local prediction log | [app.py](../app.py) |
| Bot/AI | Telegram loop + `AIAssistant` | Telegram long polling; chart/predict commands; optional Groq question answering with local data context | [api_service.py](../src/iot_digital_twin/api_service.py), [llm_service.py](../src/iot_digital_twin/llm_service.py) |
| Visualization | `viz_engine` + `video_generator` | API/Telegram heatmaps, plots, and optional daily MP4 + Hugging Face dataset upload | [viz_engine.py](../src/iot_digital_twin/viz_engine.py), [video_generator.py](../src/iot_digital_twin/video_generator.py) |
| Weather context | `weather_client` | Open-Meteo current/hourly weather, 10-minute process-local cache | [weather_client.py](../src/iot_digital_twin/weather_client.py) |
| Independent MQTT bridge | `mqtt_worker.py` | HiveMQ TLS subscribe; topic-to-node mapping; queued/batched Supabase upsert | [mqtt_worker.py](../mqtt_worker.py) |
| Shared status | `worker_state` | Atomic JSON snapshot shared by the designated worker and API processes for `/health` | [worker_state.py](../src/iot_digital_twin/worker_state.py) |
| Unity client | `UnityInferenceClient` + binder | Polls the Unity prediction route; emits update/error callbacks to TextMeshPro UI | [Unity scripts](../unity_integration/UNITY_SETUP.md) |

## Deployment topology

The committed Dockerfile uses Python 3.10, installs dependencies, copies source, and starts Supervisor. Supervisor launches five process groups within the **same container**:

~~~mermaid
flowchart TB
  Client["Browser / REST / Unity consumer"] --> NG["Nginx :7860"]
  NG -->|"/"| ST["Streamlit :8501<br/>app.py"]
  NG -->|"/api/* (prefix stripped)"| API["Uvicorn :8000<br/>2 API worker processes<br/>APP_MODE=api"]
  SUP["Supervisor"] --> NG
  SUP --> ST
  SUP --> API
  SUP --> WK["Uvicorn :8001<br/>1 worker process<br/>APP_MODE=worker"]
  SUP --> MQ["python mqtt_worker.py"]
  WK --> State[("artifacts/worker_state.json")]
  API -.->|"/health reads"| State
  API --> Files[("artifacts/lstm_checkpoint.pt")]
  WK --> Files
  ST --> Files
  MQ --> Supa[("Supabase env_readings")]
~~~

**Exact routing:** `nginx.conf` listens on port **7860**. `/api/` proxies to Uvicorn port **8000** with the `/api/` prefix removed; `/` proxies to Streamlit port **8501** with WebSocket forwarding. Local development examples instead run Uvicorn directly on 8000 and Streamlit separately on 8503. `:8001` runs the designated worker's Uvicorn process but is not exposed by Nginx.

**APP_MODE nuance:** `InferenceAPIService.__init__` starts the **background fetch thread in every process**, including `APP_MODE=api`. The mode only gates Telegram polling and alert/report/video/weather side effects, and permits only `worker`/`combined` to write the health-state file. As a result the two API processes still fetch, maintain separate caches, load/train predictors and may touch the shared checkpoint; they are **not read-only** views of a single worker-owned inference cache. The worker also performs its own fetch/model preparation. This is relevant to resource budgeting and concurrency.

**Development mode:** without `APP_MODE`, the code defaults to `combined`; a single Uvicorn instance then owns both API and notification work. Running multiple combined instances is not equivalent to the dedicated worker topology.

**Deployment automation:** [.github/workflows/hf_sync.yml](../.github/workflows/hf_sync.yml) pushes `main` to a Hugging Face Space after qualifying commits; Markdown-only changes are excluded by `paths-ignore`. This is a repository-defined workflow, not proof a Space is currently deployed.

## In-process state and persistence

| Location | Purpose | Durable / shared? |
|---|---|---|
| `InferenceAPIService._cached_clean_df` | Last validated historical frame | **Process-local memory**; independently populated per process |
| `DeepTimeSeriesPredictor` instance | Fitted weights / scalers / ordered feature names | Process-local memory, initially loaded from or saved to checkpoint |
| `artifacts/lstm_checkpoint.pt` | LSTM weights, feature schema, scaler and preprocessing configuration | Local filesystem; shared among container processes, persistence beyond container restart **not established** |
| `artifacts/autoencoder_checkpoint.pt` + `autoencoder_meta.json` | AE weights/threshold metadata | Optional local artifacts; detector may be disabled if unavailable |
| `artifacts/worker_state.json` | Latest worker heartbeat/status snapshot | Atomic local JSON file, read by API mode for `/health` |
| `artifacts/prediction_log.csv` | Dashboard forecast issued times and later ground-truth reconciliation | Local file written by Streamlit |
| `api_server.latest_command` | Remote-control last command and timestamp | **In-memory per Uvicorn worker**; not synchronized across workers or restarts |
| Supabase `env_readings` | Optional incoming MQTT rows | Remote table written by MQTT bridge only |
| Supabase `alert_logs`, `bot_logs`, `system_heartbeat` | Optional side-channel operational records | Remote tables written if Supabase credentials exist |
| Hugging Face asset dataset | Optional daily heatmap MP4 archive | Remote dataset, depends on `HF_TOKEN` |

The code does not implement a central message bus between inference/API processes, nor does it implement a Supabase-to-predictor read adapter. Filesystem checkpoints and caches should not be conflated with the remote telemetry database.

## Boundaries and trust

- **Public/external I/O:** Google Sheets CSV over HTTPS; Telegram Bot API; optional Groq and Open-Meteo; HiveMQ MQTT over TLS; optional Supabase; optional Hugging Face asset uploads.
- **API consumers:** FastAPI routes have CORS `allow_origins=["*"]` and no explicit authentication middleware in `api_server.py`. `POST /retrain` and `POST /command` are reachable like other routes unless upstream infrastructure restricts them.
- **Credentials:** loaded from environment/`.env`; do not put live secrets in diagrams or documentation. MQTT worker requires `SUPABASE_SERVICE_KEY`; the README's older `SUPABASE_KEY` example is not the worker's actual variable.
- **Not included in this repository:** ESP firmware/Home Assistant automation, hosted Google Sheet configuration, Supabase schema migrations/RLS, Hugging Face secrets/settings, the full Unity scene/client deployment.

## Source navigation

- [Telemetry and all sequences](DATA_FLOWS.md)
- [API and data schemas](API_AND_CONTRACTS.md)
- [Startup, risks, and debugging](OPERATIONS_AND_CAVEATS.md)
