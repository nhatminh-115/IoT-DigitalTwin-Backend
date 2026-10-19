# End-to-end data flows

**Source-reviewed:** 2026-10-19. Diagrams distinguish HTTP polling, asynchronous loops, shared local files, and optional network operations. Cadences are configuration values, not verified network delivery SLAs.

## 1. Primary telemetry and inference path

The repository README says an external Home Assistant automation exports sensor readings into a Google Sheet about every three minutes. That publisher is outside this repository. The implemented model pipeline begins at the **Google Sheets CSV export**, not at Supabase.

~~~mermaid
sequenceDiagram
  autonumber
  participant HA as Home Assistant (external)
  participant GS as Google Sheets CSV
  participant B as InferenceAPIService bg-fetch
  participant F as DataFetcher
  participant Q as DataQualityGate
  participant M as LSTM predictor
  participant FS as artifacts/*.pt
  participant C as Local clean DataFrame cache
  HA-->>GS: Publishes readings (~3-min per README; outside repo)
  loop Every 30 s, per process
    B->>F: fetch()
    F->>GS: HTTPS GET CSV (retry/timeout)
    GS-->>F: Timestamp + device + metric rows
    F->>F: Parse, normalize, pivot and resample 3min
    F-->>B: Wide time-series DataFrame
    B->>Q: process(df)
    Q->>Q: Flatline → outlier → interpolation checks
    Q-->>B: clean_df + quality report
    B->>M: Ensure model ready; load or retrain if needed
    M->>FS: Read or write LSTM checkpoint
    B->>C: Replace process-local cached_clean_df
  end
~~~

**Transformation detail:** `DataFetcher` identifies a timestamp column and a device identifier (for example `esp09` → `M9`), accepts common temperature/humidity/eCO2/TVOC names and optional particulate matter fields, converts readings to numbers, pivots long per-device rows to a timestamp-indexed wide frame, creates canonical names (for example `M9_Temp`, `M9_Humid`, `M9_CO2`, `M9_TVOC`), resamples to **3-minute bins using the mean**, then interpolates/fills. The alias `eCO2` is accepted and renamed to `CO2`; this is **column naming, not proof of a direct CO2 sensor**.

The `DataQualityGate` requires at least 40 rows, flags low-variance rolling windows (20 rows, epsilon 1e-4), removes z-score outliers (absolute z > 3), interpolates again, drops fully missing channels, and rejects irreparable input. The quality report is used in Streamlit; the background loop discards the report after validation.

**Model preparation:** initial checkpoint load; train when checkpoint is missing/incompatible or the feature set changes. The API service trains on the latest **1200 rows** (or enough to satisfy sequence length). The dashboard separately uses a **720-row** training tail; see Section 3.

**Failure behavior:** network, quality, or model exceptions are caught inside the background loop and stored as the last error; subsequent iterations continue. If a previously cached frame exists, it is not explicitly cleared on failure. A consumer could therefore see old cached readings, so `/health` and `last_fetch_utc` should be checked when freshness matters.

## 2. LSTM forecasting and Unity consumption

The predictor is **multivariate**: one model consumes all currently available cleaned feature channels. It differences consecutive observations, standardizes each difference channel, feeds the last **20 differenced steps** to a two-layer LSTM (hidden size 128, dropout 0.15), predicts the next standardized difference vector, then converts back to original units by adding the last observed absolute vector. `forecast_target_time` is inferred from the median positive sampling interval, falling back to 3 minutes.

~~~mermaid
sequenceDiagram
  autonumber
  participant U as UnityInferenceClient.cs
  participant N as Nginx (hosted /api)
  participant A as FastAPI /unity/predict
  participant S as InferenceAPIService
  participant M as DeepTimeSeriesPredictor
  U->>N: GET /api/unity/predict?module=M1
  N->>A: GET /unity/predict?module=M1
  A->>S: get_unity_payload(M1)
  S->>S: Read current process cached_clean_df
  S->>M: Ensure checkpoint/model readiness
  S->>M: predict_next_step(clean_df)
  M-->>S: Next-step prediction for all columns
  S->>S: Filter M1, compute current/predicted/delta
  S-->>U: JSON: module, times, feature objects
  U->>U: JsonUtility.FromJson + OnPredictionUpdated
  U->>U: UI binder updates TextMeshPro
  Note over U: Client default poll interval = 180 s
~~~

When using local port `8000` without Nginx, Unity requests `/unity/predict` directly; when using published Nginx port `7860`, the client base URL must include `/api`. `UnityInferenceClient.cs` defaults to localhost:8000 and polls every **180 seconds**; its UI binder subscribes to success/error events. The code also exposes `/unity/predict/all` so a different consumer can obtain all modules in one request.

The handler is not a server-pushed real-time stream: **Unity pulls a snapshot**. The service takes its latest in-process clean data and computes on-demand prediction; it does not fetch MQTT/Supabase records on the Unity request path. For the exact object shape see [API contracts](API_AND_CONTRACTS.md#unity-prediction-contract).

## 3. Independent Streamlit dashboard path

The dashboard is **not a presentation-only frontend to the FastAPI service**. It directly constructs its own fetcher, quality gate, evaluator and session-state predictor. On each Streamlit run/refresh (default **180 seconds**, adjustable), it fetches the CSV, cleans data, loads or trains its own model, predicts one next step, generates historical MC-dropout forecasts, computes residual/MAE/RMSE charts and stores a prediction log.

~~~mermaid
flowchart LR
  Browser["Browser"] --> Streamlit["app.py / Streamlit :8501 hosted<br/>or :8503 local"]
  Streamlit --> GS["Google Sheets CSV"]
  Streamlit --> Gate["DataFetcher + QualityGate"]
  Gate --> P["Dashboard predictor<br/>(session state)"]
  P <--> PT[("artifacts/lstm_checkpoint.pt")]
  P --> Eval["ModelEvaluator + MC dropout"]
  P --> Log[("artifacts/prediction_log.csv")]
  Eval --> Browser
  Log --> Browser
~~~

The dashboard trains from a tail of **720 rows**, whereas the API service uses **1200 rows**; both may read/write the same checkpoint path. The local log appends predicted values and later reconciles them with observed ground truth. Its `/` URL is routed to Streamlit by Nginx; API traffic goes to `/api/`. The dashboard weather strip independently obtains weather context.

## 4. Telegram command, AI and alert paths

Only a process running `APP_MODE=combined` or `APP_MODE=worker` starts the Telegram long-polling thread. Incoming `getUpdates` responses are dispatched to status, chart, heatmap, rank, comparison, forecast, video or `/ask` handlers. Most commands use the in-memory clean data frame rather than a database query.

~~~mermaid
sequenceDiagram
  autonumber
  participant Chat as Telegram user
  participant TG as Telegram Bot API
  participant Worker as Bot polling thread
  participant Cache as Worker cached_clean_df
  participant LL as AIAssistant / SheetContextFetcher
  participant Groq as Groq API (optional)
  participant DB as Supabase logs (optional)
  Chat->>TG: /ask What is M1 temperature?
  Worker->>TG: getUpdates (long poll)
  TG-->>Worker: New text + chat/user details
  Worker->>LL: answer(query)
  LL->>Cache: get_df() for recent readings
  LL->>LL: Detect intent, summarize status/trends/alerts
  opt Semantic cache miss
    LL->>Groq: Chat completion with structured local context
    Groq-->>LL: Answer text and usage
  end
  LL-->>Worker: Answer, latency, cache metadata
  Worker->>TG: sendMessage(answer)
  Worker-->>DB: Optional bot_logs insert
  TG-->>Chat: Answer
~~~

`AIAssistant` uses Groq's `llama-3.3-70b-versatile` if `GROQ_API_KEY` is configured. Before calling the model, `SheetContextFetcher` builds bounded context from the **worker process's cached clean DataFrame**, detects current/trend/today/module-status intent, checks active/frozen nodes, and formats a recent data table. `SemanticCache` keeps fuzzy-matched answers for five minutes. The implementation does **not** perform retrieval from the Supabase `env_readings` table.

**Automated alerts:** every successful 30-second fetch cycle in `worker`/`combined` mode invokes threshold checks, hourly reporting, AE scoring (if detector loaded), daily video generation and outdoor-weather sanity checks. Core thresholds in `api_service.py`: Temp <18 or >33 °C; Humid <30 or >75%; CO2 >1200; TVOC >300. Individual sensor alerts are buffered for up to 120 seconds, rate-limited for one hour and periodically re-alerted if persistent; hysteresis reduces flapping. The AE scorer compares reconstruction error to its **checkpoint metadata threshold**, not a universally fixed hard-coded cutoff. Messages are sent through Telegram; qualifying records can also be inserted asynchronously into Supabase `alert_logs`.

**Other auxiliary jobs:** an hourly summary near the beginning of each hour; daily heatmap MP4 for the previous day at approximately 00:10–00:14 ICT (re-fetches CSV and resamples at 30 minutes); an optional asset upload to Hugging Face; Open-Meteo comparisons for configured outdoor/semi-outdoor nodes (10-minute weather cache, per-node drift-alert cooldown four hours); a periodic `system_heartbeat` insert approximately every five minutes, if Supabase logging is configured.

## 5. Independent MQTT to Supabase ingestion

This is a **separate ingestion pipeline**, not a replacement model feed. `mqtt_worker.py` reads broker credentials and a Supabase **service key** at startup. It subscribes to `esp/+` at QoS 1 over TLS, parses JSON from known ESP topics, converts measurements, adds a server-side UTC timestamp, queues records in memory, then periodically upserts batches into `env_readings`.

~~~mermaid
sequenceDiagram
  autonumber
  participant ESP as ESP sensor (outside repo)
  participant Broker as HiveMQ Cloud
  participant MQTT as mqtt_worker.py
  participant Q as In-memory record_queue
  participant DB as Supabase env_readings
  ESP->>Broker: PUBLISH esp/esp09 JSON (device firmware outside repo)
  Broker-->>MQTT: MQTT subscription esp/+ (QoS 1, TLS)
  MQTT->>MQTT: Parse topic + JSON; map esp09 → M9
  MQTT->>MQTT: Stamp ts = server receipt time (UTC)
  MQTT->>Q: enqueue parsed {ts,node_id,temp,humidity,co2,tvoc}
  loop Every ~30 s or queue reaches 100
    Q->>MQTT: drain up to 100 records
    MQTT->>DB: upsert(batch, conflict ts,node_id)
    alt Request fails
      MQTT->>Q: Best-effort requeue; may drop if full
    end
  end
~~~

Eight explicit mappings are supported: `esp01→M1`, `esp04→M4`, `esp06→M6`, `esp07→M7`, `esp08→M8`, `esp09→M9`, `esp10→M10`, `esp11→M11`. Example fields are `t1` (Temp), `h1` (Humidity), `co2` and `tvoc`. Unknown devices/bad JSON are skipped; nonnumeric values become null. Default queue maximum is 10,000 records; batch size 100; flush interval 30 seconds. The worker attempts a final drain at shutdown. MQTT QoS 1 provides broker-level delivery semantics, but the in-memory queue/retry implementation does **not** guarantee end-to-end exactly-once or lossless persistence. `ts` represents worker receipt, not necessarily physical measurement time.

**Do not draw an edge** from Supabase `env_readings` back to the API/ML pipeline: no such read operation is present in `api_service.py`, `data_fetcher.py` or `app.py`.

## 6. HTTP command and remote-control dashboard

`remote_control_dashboard.html` embeds a video stream and sends `POST /api/command` with `{"metric": "..."}` through Nginx, plus `GET /api/command` for a connectivity check. In `api_server.py` the POST handler writes a module-level Python dictionary `latest_command` with a metric and server timestamp; GET returns it.

~~~mermaid
sequenceDiagram
  participant UI as Remote control HTML
  participant N as Nginx /api/command
  participant A as FastAPI worker process
  participant C as latest_command (process memory)
  UI->>N: POST /api/command {metric: "temp"}
  N->>A: POST /command
  A->>C: Replace dict with metric + timestamp
  A-->>UI: {status:"success", command:{...}}
  UI->>N: GET /api/command
  N->>A: GET /command
  A-->>UI: latest_command (of worker reached)
~~~

**Important limitation:** Supervisor starts **two** Uvicorn API processes. They have independent in-memory `latest_command` values; consecutive POST and GET requests can reach different workers and disagree, and commands disappear on process restart. The repository demonstrates a producer endpoint and UI, but does not demonstrate a durable, cross-process command consumer or command-delivery acknowledgement. A consumer in another repository must not treat POST success as proof that a visualizer executed the requested change.

## 7. Data-format boundaries and timing

| Boundary | Transport | Main data shape | Nominal timing |
|---|---|---|---|
| Home Assistant → Google Sheets | External integration, per README | Row records with device/time/metrics | ~3 min (README claim) |
| Sheet → API background loops | HTTPS GET CSV | Long table → timestamp × canonical wide columns | Every 30 s per process |
| Sheet → Streamlit | HTTPS GET CSV | Same normalizer / quality gate | Default refresh every 180 s |
| API → Unity | HTTP JSON, request/response | `module`, issue/target time, current/predicted/delta/features | Unity poll every 180 s |
| Telegram → backend | Bot API HTTP long polling | Update/message/callback objects | API polling timeout 20 s |
| Groq / Open-Meteo | HTTPS | Contextual LLM output / weather payload | On ask / 10-minute weather cache |
| MQTT broker → bridge | MQTT TLS | Topic `esp/+` and JSON telemetry | On publish |
| MQTT bridge → Supabase | HTTPS DB upsert | `env_readings` row batches | Every 30 s or 100 records |
| Worker → API status | Atomic local JSON file | `worker_state.json` | On successful fetch cycle |
| Control UI → API | HTTP POST and GET | Last `metric` command + timestamp | On user interaction |

The **telemetry sampling interval**, **background fetch interval**, **dashboard refresh**, and **Unity polling period** are distinct. Updating the 30-second backend fetch interval does not change the sensor's sampling rate or Unity polling schedule.

## 8. Reproduction checklist

For code readers wanting to verify the diagrams:

1. Track `InferenceAPIService.__init__` and `_background_fetch_loop`, then `_load_latest_clean_data` in [api_service.py](../src/iot_digital_twin/api_service.py).
2. Track `DataFetcher._normalize_schema` and `DataQualityGate.process` for input semantics.
3. Track `DeepTimeSeriesPredictor.fit` / `predict_next_step` and `api_server.py` routes.
4. Inspect [app.py](../app.py) to confirm Streamlit's independent fetch path.
5. Inspect [mqtt_worker.py](../mqtt_worker.py) to confirm Supabase writes and absence of inference reads.
6. Inspect [nginx.conf](../nginx.conf) and [supervisord.conf](../supervisord.conf) for actual hosted processes.
