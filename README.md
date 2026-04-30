---
title: IoT Digital Twin Dashboard
emoji: 🏭
colorFrom: indigo
colorTo: blue
sdk: docker
app_file: app.py
pinned: false
---

# IoT Digital Twin Backend & Forecast Dashboard

Backend service, real-time forecasting dashboard, and AI assistant for the UEH Campus V IoT Digital Twin project. Combines LSTM forecasting, anomaly detection, Telegram bot integration, and AI-powered insights.

**Live Components:**
- Streamlit dashboard (web UI) on port 8503
- FastAPI inference API on port 8000
- MQTT worker → Supabase integration
- Telegram bot command handler

## Architecture Overview

```
Main runtime path

HomeAssistant (3-min cadence)
  |
  v
Google Sheets (CSV export)
  |
  v
DataFetcher -> DataQualityGate -> InferenceAPIService (background loop every 30s)
             |
             +-> LSTM Predictor -> FastAPI endpoints (/latest, /predict/latest, /unity/*)
             +-> Autoencoder -> anomaly alerts
             +-> Telegram polling -> commands (/chart, /predict, /ask, ...)
             |        |
             |        +-> AIAssistant (Groq) using cached clean DataFrame context
             +-> Hourly report + daily video + weather sanity checks
             +-> Optional Supabase logs (alert_logs, bot_logs, system_heartbeat)

Parallel web UI path

Google Sheets -> DataFetcher -> DataQualityGate -> Streamlit app (app.py, port 8503)
                 -> local predictor/evaluator for dashboard charts

Optional ingestion path (independent)

HiveMQ MQTT (esp/+) -> mqtt_worker.py -> Supabase table env_readings
```

Note: current forecasting/alert API flow uses Google Sheets as the primary data source. MQTT -> Supabase is currently an optional side pipeline and is not wired into InferenceAPIService fetch logic.

## Components

### Data Pipeline

| Component | File | Role |
|---|---|---|
| `DataFetcher` | `data_fetcher.py` | Fetches CSV from Google Sheets, normalizes column schema to `M#_Temp/Humid/CO2/TVOC` |
| `DataQualityGate` | `data_quality_gate.py` | Flatline detection (rolling variance, 20-row window ~60 min), z-score outlier removal, linear interpolation |
| `InferenceAPIService` | `api_service.py` | Background fetch loop (30s), alert dispatch, hourly reports, Telegram polling |

### ML Models

| Model | Details |
|---|---|
| **LSTM Forecaster** | 2-layer LSTM, sequence length 20 (~60 min), first-order differencing + z-score normalization, one-step-ahead prediction |
| **Autoencoder** | Reconstruction-error anomaly detection, soft alert layer complementing rule-based thresholds |

Training window: last **1200 rows (~60h)** at 3-min cadence.

### Sensor Thresholds

| Metric | Low | High |
|---|---|---|
| Temperature | < 18 °C | > 33 °C |
| Humidity | < 30 % | > 75 % |
| CO2 | — | > 1200 ppm |
| TVOC | — | > 300 ppb |

### Telegram Bot Commands

| Command | Description |
|---|---|
| `/getcurrent_detail` | Full sensor table, all 8 nodes |
| `/getcurrent_short` | Summary: avg temp/humidity + anomaly count |
| `/getcurrent_alert` | Active threshold breaches only |
| `/ask <question>` | AI assistant. Usage: /ask <question> |
| `/chart` | Interactive chart: select range, node, and metric |
| `/heatmap` | Interactive spatial heatmap: select metric |
| `/rank` | Interactive node ranking: select metric |
| `/compare` | Interactive node comparison: select two nodes |
| `/predict` | Interactive LSTM forecast: select node and metric |

Power users can also use the full command syntax directly (e.g. `/chart_day_all_temp`, `/chart_hour_M1_temp`, `/heatmap_co2`, `/rank_humid`, `/compare_M1_M4`, `/predict_M7_tvoc`).

Command parameter order: **range → node → metric** (omitted where not applicable).

Nodes: `M1 (Canteen Garden)`, `M4 (Studio ISCM)`, `M6 (ISCM Staircase)`, `M7 (Sky Garden)`, `M8 (ISCM Balcony)`, `M9 (Hotel Kitchen)`, `M10 (Hotel Corridor)`, `M11 (Hotel Balcony)`

Alert cooldown: 1 hour per sensor channel.

### AI Assistant (`/ask`)

Powered by `llama-3.3-70b-versatile` via Groq API. Data source is the in-memory Google Sheets DataFrame — no additional database required.

**Context sent to LLM per query:**

```
=== SUMMARY ===
Latest data    : 11/04/2026 00:30 ICT
Active nodes   : M1, M4, M6, M7, M8, M9, M10, M11 (8/8)
Silent nodes   : none
Frozen sensors : none (unchanged >=2h / 40 readings)
Alerts         : none
================

=== DATA TABLE: Latest readings (~30 min) ===
...
```

**Intent detection (Python-side, zero LLM tokens):**

| Intent | Trigger keywords | Data window |
|---|---|---|
| `module_status` | "dead", "offline", "which module", "status" | 5 rows (~15 min) |
| `current` | _(default)_ | 10 rows (~30 min) |
| `trend` | "trend", "this morning", "6h", "past hour" | 120 rows (~6h) |
| `today` | "today", "all day", "24h" | 480 rows (~24h) |

**Dead node detection:** node is marked silent if no reading in the last **60 minutes**.

**Frozen sensor detection:** value unchanged for **40 consecutive readings (~2h)** → flagged in SUMMARY.

**Semantic cache:** fuzzy match (thefuzz, 80% threshold), 5-minute TTL.

## Cadence Reference (3-min sampling)

| Window | Rows |
|---|---|
| 1 hour | 20 |
| 1 day | 480 |
| 7 days | 3360 |
| Training data | 1200 (~60h) |
| Flatline detection | 20 (~60 min) |
| Frozen sensor | 40 (~2h) |

---

## Setup & Installation

### Prerequisites
- Python 3.11+
- 4GB RAM (minimum for model training)
- Internet connection (Google Sheets API, Groq API, Telegram API)

### 1. Clone and install dependencies

```bash
git clone <repo-url>
cd iot_backend/Model_DTW_1
pip install -r requirements.txt
```

### 2. Configure environment

Create `.env` file in project root:

```bash
# Telegram
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here

# Groq API (for /ask command)
GROQ_API_KEY=your_groq_api_key_here

# Optional: Supabase (not currently used)
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-supabase-key

# Data source (default Google Sheets URL already configured)
# CSV_URL=https://docs.google.com/spreadsheets/d/...
```

### 3. Run services

**Terminal 1 - Dashboard:**
```powershell
python3.11.exe -m streamlit run app.py --server.port 8503
```
Open browser: `http://localhost:8503`

**Terminal 2 - FastAPI:**
```powershell
python3.11.exe -m uvicorn api_server:app --host 0.0.0.0 --port 8000
```
API available at: `http://localhost:8000`

**Terminal 3 - Telegram Bot (optional):**
```powershell
python3.11.exe api_server.py  # Bot polling is handled in api_service.py background loop
```

---

## Quick Start Examples

### 1. Get latest sensor readings

```bash
curl http://localhost:8000/latest
```

### 2. Get LSTM forecast for next step

```bash
curl http://localhost:8000/predict
# Optional: single module
curl "http://localhost:8000/predict?module=M1"
```

### 3. Get hourly history

```bash
curl http://localhost:8000/latest/hour
```

### 4. Force model retraining

```bash
curl -X POST http://localhost:8000/retrain
```

### 5. Telegram bot commands (chat with bot)

- `/getcurrent_short` - Quick summary
- `/predict_M1_temp` - Forecast M1 temperature
- `/chart_day_all_temp` - Daily temperature chart
- `/ask What is the average temperature?` - AI query

---

## Project Structure

```
src/iot_digital_twin/
├── __init__.py
├── data_fetcher.py          # Fetch & parse Google Sheets CSV
├── data_quality_gate.py     # Outlier removal, interpolation, flatline detection
├── predictor.py             # LSTM model training & inference
├── model_evaluator.py       # Training pipeline & checkpoint management
├── anomaly_detector.py      # Autoencoder for reconstruction error
├── api_service.py           # FastAPI service + Telegram bot loop
├── llm_service.py           # Groq LLM integration & semantic cache
├── weather_client.py        # Weather API client (optional)
├── viz_engine.py            # Chart/visualization generation
└── worker_state.py          # Persistent state management

artifacts/
├── lstm_checkpoint.pt       # Trained LSTM weights
├── autoencoder_checkpoint.pt # Trained autoencoder
└── prediction_log.csv       # Historical predictions

config/
└── outdoor_nodes.json       # Node metadata & coordinates

tests/
├── conftest.py              # Test fixtures
└── test_*.py                # Unit & integration tests
```

---

## Model Details

### LSTM Predictor

- **Architecture:** 2-layer LSTM with dropout, fully connected output layer
- **Input:** Last 20 readings (3-min cadence = ~60 min history)
- **Processing:** First-order differencing + z-score normalization
- **Output:** Next-step (3-min) prediction per metric
- **Training:** Last 1200 rows (~60h) from CSV
- **Loss:** MSE with regularization

### Autoencoder Anomaly Detector

- **Architecture:** 4-layer encoder-decoder network
- **Input:** Normalized sensor readings
- **Threshold:** 2x median reconstruction error
- **Use:** Soft anomaly alerts (complements hard threshold rules)

---

## API Response Format

### `/latest` (Legacy Unity format)
```json
{
  "timestamp": "2026-04-30T10:30:00Z",
  "modules": {
    "M1": {"temp": 26.5, "humid": 62.3, "co2": 450, "tvoc": 15},
    "M4": {"temp": 27.1, "humid": 58.2, "co2": 480, "tvoc": 20},
    ...
  }
}
```

### `/predict` (Forecast)
```json
{
  "timestamp": "2026-04-30T10:30:00Z",
  "forecast_step": "next_3min",
  "predictions": {
    "M1": {"temp": 26.7, "humid": 62.1, "co2": 452, "tvoc": 14},
    ...
  },
  "confidence_metrics": {
    "mae_temp": 0.45,
    "mae_humid": 1.2,
    ...
  }
}
```

---

## Troubleshooting

### Dashboard won't start (port 8503 error)

```bash
# Kill existing process
netstat -ano | findstr :8503
taskkill /PID <PID> /F

# Or use different port
streamlit run app.py --server.port 8504
```

### API returns 503 error (model not ready)

This means the model is still training. Wait 2-3 minutes and retry.

```bash
# Check status
curl http://localhost:8000/health
```

### CSV fetch timeout

Increase timeout in `data_fetcher.py` or check internet connection:

```python
# Default: 30 seconds
response = requests.get(CSV_URL, timeout=60)
```

### Telegram bot not responding

1. Verify token: `curl "https://api.telegram.org/bot<YOUR_TOKEN>/getMe"`
2. Check chat ID matches
3. Ensure `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` in `.env`
4. Restart API server

### Out of memory (CUDA/CPU)

Model uses ~2GB during training. If OOM:

```python
# In predictor.py, reduce batch size or sequence length
SEQUENCE_LENGTH = 10  # default: 20
BATCH_SIZE = 8        # default: 16
```

### Model checkpoint corrupt

Delete and retrain:

```bash
rm artifacts/lstm_checkpoint.pt
curl -X POST http://localhost:8000/retrain
```

---

## Development & Testing

### Run tests

```bash
# All tests
pytest tests/

# Unit only
pytest tests/unit/

# Integration only (requires external services)
pytest tests/ -m integration

# With coverage
pytest tests/ --cov=src --cov-report=html
```

### Local development

1. Modify code in `src/iot_digital_twin/`
2. API auto-reloads: `uvicorn api_server:app --reload`
3. Dashboard auto-reloads on file save
4. Check logs in terminal

### Add new endpoint

```python
# In api_server.py
@app.get("/my-endpoint")
async def my_endpoint(service: InferenceAPIService = Depends(get_service)):
    return await service.my_method()
```

---

## Performance Notes

- **Data fetch:** ~2-5 seconds (Google Sheets API)
- **Quality gate:** ~0.5 seconds (outlier + interpolation)
- **LSTM inference:** ~0.2 seconds (per module)
- **Model retraining:** ~30-60 seconds (1200 rows, 2-layer LSTM)
- **LLM query:** ~3-5 seconds (Groq API)

**Resource usage (steady state):**
- Memory: ~800MB (data + model in RAM)
- CPU: <5% (async, polling every 30s)
- Network: ~100KB/s (polling Google Sheets + Telegram)

---

## MQTT Integration (Optional)

If using MQTT data source (HiveMQ Cloud → Supabase):

```bash
python mqtt_worker.py
```

Requires `.env`:
```
MQTT_BROKER=your-hivemq-broker.com
MQTT_USER=username
MQTT_PASS=password
MQTT_TOPIC=esp/+
SUPABASE_URL=...
SUPABASE_KEY=...
```

---

## Contributing

1. Create feature branch: `git checkout -b feature/my-feature`
2. Add tests in `tests/`
3. Run `pytest` locally
4. Commit & push
5. Submit PR with description

---

## License & Contact

For issues or questions, contact the UEH IoT team or submit issues on the project repository.

Last updated: April 2026
