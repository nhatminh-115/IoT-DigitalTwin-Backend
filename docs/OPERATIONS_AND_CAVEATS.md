# Operations, configuration, observability and known caveats

**Based on checked-in behavior as of 2026-10-19.** This document explains how to run and debug the existing implementation; hardening suggestions are explicitly marked as such and are **not claims about implemented features**.

## Run modes

### Local development

From repository root after installing dependencies from `requirements.txt`:

```bash
# Terminal 1: inference API and combined bot/alert loops (default APP_MODE)
python -m uvicorn api_server:app --host 0.0.0.0 --port 8000

# Terminal 2: independent dashboard
python -m streamlit run app.py --server.port 8503

# Optional, separate MQTT bridge; requires HiveMQ and Supabase credentials
python mqtt_worker.py
```

Direct API examples:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/latest
curl "http://localhost:8000/predict/latest?module=M1"
curl "http://localhost:8000/unity/predict?module=M1"
curl http://localhost:8000/unity/predict/all
```

`APP_MODE` defaults to `combined`. If running more than one local API instance with `combined`, each can start its own Telegram polling and side-effect loops; this is not the intended single-owner production layout.

### Containerized process supervisor

[Dockerfile](../Dockerfile) uses `python:3.10-slim`, exposes port 7860 and invokes [supervisord.conf](../supervisord.conf). The Supervisor program list is:

| Program | Command / bind | Mode / responsibility |
|---|---|---|
| `nginx` | Reverse proxy on `:7860` | `/` → Streamlit, `/api/` → FastAPI |
| `api` | `uvicorn api_server:app --host 0.0.0.0 --port 8000 --workers 2` | `APP_MODE=api` (still starts background fetch in each process) |
| `worker` | `uvicorn api_server:app --host 127.0.0.1 --port 8001` | `APP_MODE=worker`; owner of Telegram, alerts, reports, worker health file |
| `dashboard` | `streamlit run app.py --server.port 8501` | Independent Streamlit model/dashboard pipeline |
| `mqtt_worker` | `python mqtt_worker.py` | Always started by Supervisor; independent MQTT ingestion |

Port `8001` is used to keep a worker process alive and import the service; it is not Nginx-exposed. From outside the container use `http://<host>:7860/api/health`, not `http://<host>:7860/health`.

**Deployment caveat:** the MQTT worker reads required credentials at import time. If missing, that supervised child process will fail/restart even if the main API/dashboard could otherwise work. The optionality of MQTT is logical, **not conditional in the current Supervisor config**.

### Hugging Face sync

[.github/workflows/hf_sync.yml](../.github/workflows/hf_sync.yml) pushes updates of `main` to a Hugging Face Space if its token secret is configured. Markdown-only commits (including the docs here) are excluded from automatic sync by `'**.md'`. Manual workflow dispatch exists. A workflow configuration is not evidence of a currently healthy hosted deployment.

## Configuration reference

| Setting | Used by | Required / behavior |
|---|---|---|
| `APP_MODE` | `api_service.py` | Default `combined`; configured as `api` or `worker` in Supervisor |
| `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` | Telegram thread, alerts/reports | Optional; without both, Telegram polling waits rather than entering an active poll |
| `TELEGRAM_API_URL` | Telegram calls | Optional override, default `https://api.telegram.org` |
| `GROQ_API_KEY` | `AIAssistant` | Optional; missing key disables `/ask` AI integration |
| `SUPABASE_URL` | API logger, MQTT worker | Optional for API logging; required for MQTT worker |
| `SUPABASE_SERVICE_KEY` | API logger, MQTT worker | Optional for API logging; **required for MQTT worker** |
| `HIVEMQ_HOST`, `HIVEMQ_USER`, `HIVEMQ_PASS` | MQTT worker | Required when MQTT process starts |
| `HIVEMQ_PORT` | MQTT worker | Default `8883` with TLS |
| `HIVEMQ_CLIENTID` | MQTT worker | Default `supabase-worker-01` |
| `FLUSH_INTERVAL_SEC` | MQTT worker | Default `30` |
| `FLUSH_BATCH_SIZE` | MQTT worker | Default `100` |
| `QUEUE_MAXSIZE` | MQTT worker | Default `10000` |
| `WORKER_STATE_PATH` | `worker_state.py` | Optional; defaults to `artifacts/worker_state.json` |
| `HF_TOKEN` | Video asset upload and workflow secret | Optional for runtime video upload; configured as an Actions secret for CI sync |
| `config/outdoor_nodes.json` | Weather sanity checks | Optional file with campus coordinate/node thresholds; if missing, weather checks skip |
| `artifacts/autoencoder_checkpoint.pt` + `autoencoder_meta.json` | AE anomaly detection | Optional pair; detector initialization failure is non-fatal |

**CSV configuration gotcha:** `api_server.py`, `app.py` and `video_generator.py` each define a Google Sheets `CSV_URL` constant in source; this implementation does **not** read a uniform `CSV_URL` environment variable to override them. The existing README shows a commented example environment variable, but setting it alone does not rewire the code. If the source Sheet changes, inspect all three entry points.

**Credential mismatch with older README:** the MQTT worker uses `HIVEMQ_HOST` / `HIVEMQ_USER` / `HIVEMQ_PASS` and `SUPABASE_SERVICE_KEY`, not `MQTT_BROKER` / `MQTT_USER` / `MQTT_PASS` and `SUPABASE_KEY`. Use names from actual code.

**Never commit** real Telegram tokens, Groq keys, Supabase service keys, broker credentials, or HF tokens to the repository. The Supabase service-role key is especially sensitive; it should be held server-side only.

## Health and observability

### What /health actually tells you

- In `APP_MODE=worker` or `combined`, the endpoint reads local service/thread/cache state. In `APP_MODE=api`, it reads the worker's atomic `worker_state.json` snapshot instead.
- On successful worker fetch cycles the worker writes `last_fetch_utc`, `last_error`, `cache_rows`, `model_fitted`, and thread liveness values.
- Worker snapshot age 300 seconds or more is **degraded** (or already degraded if an error is recorded); age greater than 900 seconds or a reported stopped background thread is **unhealthy**.
- This is a status snapshot of the worker, not a cross-process transaction or end-to-end sensor freshness probe. It does not measure the time of the most recent **source sensor reading**.
- `trace_id` on FastAPI error envelopes assists log correlation, but there is no dedicated distributed tracing integration in this repository.

Suggested checks for an incident:

```bash
curl -sS http://localhost:8000/health
curl -sS http://localhost:8000/latest
curl -sS "http://localhost:8000/unity/predict?module=M1"
curl -sS http://localhost:8000/command
```

In a hosted environment, substitute `http://<host>:7860/api` for `http://localhost:8000`. Compare `last_fetch_utc`, response timestamps and the expected sensor update cadence; a green health status alone does not guarantee newest telemetry.

### What is logged

- API/worker: Python logs for errors, fetch/model failures and Telegram failures.
- Optional Supabase: `alert_logs` (threshold/AE events), `bot_logs` (AI question/response metadata), `system_heartbeat` (roughly once every 10 30-second iterations, with 3-day deletion attempted).
- MQTT bridge: console logs for connect, parsing errors, queue overflow and batch upsert failures.
- Streamlit: dashboard UI and local `artifacts/prediction_log.csv` (issued predictions and later observed targets).
- No general durable API request audit log or central job queue is implemented.

## Model lifecycle

The service's `_ensure_model_ready` initially attempts to load `artifacts/lstm_checkpoint.pt`; if missing/incompatible/not fitted or if feature columns differ, it trains on the last **1200** clean rows. The dashboard can also load/train the same checkpoint using **720** rows. Model settings in [predictor.py](../src/iot_digital_twin/predictor.py): 20-step window, two LSTM layers, 128 hidden units, dropout 0.15, 40 epochs, Adam and MSE objective. Forecast outputs are first-difference predictions reconstructed in original sensor units.

The API `POST /retrain` forces retraining using the calling API process's cache, then writes the checkpoint. It is **synchronous** and not access-controlled by a FastAPI auth dependency. It is not an asynchronous training-job API or coordinated single-writer model registry.

AE inference depends on two compatible pre-existing artifacts; the runtime only loads them. No automatic AE retraining workflow is visible in `api_service.py`. The runtime threshold comes from saved metadata.

## Known implementation caveats

These items are **observed from source**; the suggested remediations are future work.

| Issue | Observed implementation | Consequence / future remediation |
|---|---|---|
| Two sources are not unified | `data_fetcher.py` reads Sheet CSV; MQTT bridge only writes `env_readings` | No MQTT→LSTM failover; introduce an explicit source adapter and documented reconciliation before calling MQTT the primary feed |
| Process-per-worker background fetch | Background thread unconditionally starts in `InferenceAPIService.__init__`, even when `APP_MODE=api` | 2 API workers + worker independently fetch, prepare models/cache and may overwrite checkpoint; consider assigning a single ingestion/training owner or a shared inference service |
| Checkpoint write coordination | Local `lstm_checkpoint.pt` is shared by dashboard and multiple API processes; no cross-process file lock/atomic model-version promotion is evident | Potential concurrent writes/races or inconsistent model versions; add model lifecycle coordination |
| Volatile command store | `latest_command` is a module global in each API process | With `--workers 2`, POST and GET can disagree; use shared DB/Redis queue or a single command process with delivery acknowledgements |
| Command not end-to-end acknowledged | `POST /command` confirms memory write only; no consumer in this repo acknowledges command execution | Do not show "command executed" based only on POST success |
| Limited endpoint access control | No route authentication; broad CORS; `POST /retrain` and `POST /command` exposed | Add authz/authn, rate limits, and upstream network restrictions before opening to untrusted clients |
| Per-process cache vs health | API reads worker-state file for `/health` but predictions use each API process's own cache | Health can be good while a specific API worker is warming up; expose per-instance readiness/freshness |
| Volatile MQTT buffer | `queue.Queue`, best-effort requeue on failures; queue overflow drops; server receipt timestamp | No durable spool, source measurement timestamp or exactly-once delivery; use durable broker persistence/idempotency strategy if needed |
| Legacy history format | `created` is time-only with one outer `date` | Multi-day rows can lose date identity; add a versioned timestamped history contract |
| Data semantics | Input `eCO2` aliases to output `CO2` | Preserve sensor provenance and differentiate directly measured CO2 from calculated eCO2 |
| Optional job side effects | Telegram, Groq, weather, HF asset publishing and Supabase logging depend on secrets/network | Failure of optional integrations should be monitored separately; not a guarantee of telemetry/model availability |
| Runtime filesystem persistence | Checkpoints, worker state, generated videos and prediction log use container-local paths | External volume/durable backing is not specified by Dockerfile; verify hosting retention rules |
| Timestamp assumptions | CSV parsing uses `dayfirst=True` and a timezone-naive DataFrame, while MQTT stamps UTC receipt | Cross-pipeline timeline comparison needs explicit timezone and event-time normalization |

## Incident-oriented troubleshooting

**503 on forecast/latest:** consult `/health`, check whether the API process's `_cached_clean_df` is populated, inspect Google Sheet network access/CSV parse errors and model checkpoint training. A healthy worker-state report can coexist with a warming API process.

**Data quality rejection:** verify at least 40 aligned records and enough valid numeric columns. Flatline masks and z-score outlier filtering can remove values before interpolation; inspect `DataQualityGate` rather than assuming missing values came from the original CSV.

**Telegram silence:** check token/chat ID and whether exactly one worker or combined instance owns `getUpdates`. Verify that `APP_MODE=api` instances are intentionally not polling Telegram.

**MQTT worker restarts:** check `HIVEMQ_*` and `SUPABASE_SERVICE_KEY`; Supervisor starts this program unconditionally. Inspect subscription `esp/+`, known device mapping and queue/backpressure logs. No successful MQTT batch implies that LSTM forecasts are broken only if a separate (not present here) integration depends on the bridge.

**Remote control appears inconsistent:** verify which Uvicorn worker handled `POST /command` and `GET /command`. This is not solved by polling more frequently because command state is per-process.

**Wrong data source after editing .env:** the Google Sheets URLs are defined in three Python files, rather than uniformly injected through environment variables.

## Validation scope

This documentation was derived from repository files, routes, configs and checked-in tests. It was **not** based on an execution against live Home Assistant, Google Sheets, HiveMQ, Supabase, Telegram, Hugging Face, or a running Unity scene. Existing [API tests](../tests/test_api_endpoints.py) cover selected route/error shapes but do not establish deployed network availability or data delivery guarantees.

See [documentation index](README.md), [architecture](SYSTEM_ARCHITECTURE.md), [end-to-end flows](DATA_FLOWS.md) and [API contracts](API_AND_CONTRACTS.md).
