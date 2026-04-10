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

Backend service, real-time forecasting dashboard, and AI assistant for the UEH Campus V IoT Digital Twin project.

## Architecture Overview

```
HomeAssistant (3-min cadence)
        |
        v
Google Sheets (CSV export)
        |
        v
DataFetcher --> DataQualityGate --> InferenceAPIService
                                         |
                      +------------------+------------------+
                      |                  |                  |
                 LSTM Predictor    AnomalyDetector    Telegram Bot
                 (forecasting)     (autoencoder)      (/ask AI)
                      |                  |                  |
                  FastAPI            Alerts           AIAssistant
                  endpoints                          (Groq LLM)
```

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
| `/getcurrent_detail` | Full node table with all 8 sensors |
| `/getcurrent_short` | Average temp/humidity + active anomaly count |
| `/getcurrent_alert` | Active threshold breaches only |
| `/ask <question>` | AI assistant powered by Groq LLM (Vietnamese) |

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
| `module_status` | "chết", "offline", "module nào", "trạng thái" | 5 rows (~15 min) |
| `current` | _(default)_ | 10 rows (~30 min) |
| `trend` | "xu hướng", "sáng nay", "6h", "giờ qua" | 120 rows (~6h) |
| `today` | "hôm nay", "cả ngày", "24h" | 480 rows (~24h) |

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

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `TELEGRAM_BOT_TOKEN` | Yes | Telegram bot token |
| `TELEGRAM_CHAT_ID` | Yes | Default chat ID for alerts |
| `GROQ_API_KEY` | Yes (AI) | Groq API key for `/ask` command |
| `SUPABASE_URL` | No | Not used by current pipeline |
| `SUPABASE_KEY` | No | Not used by current pipeline |

## Getting Started

```bash
pip install -r requirements.txt
streamlit run app.py          # Dashboard
uvicorn api_server:app --reload   # API Server
```

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Service health and model status |
| `GET` | `/latest` | Latest sensor row (Unity legacy format) |
| `GET` | `/latest/hour` | Last 20 rows (~1h) |
| `GET` | `/latest/day` | Last 480 rows (~24h) |
| `GET` | `/latest/week` | Last 3360 rows (~7d) |
| `GET` | `/predict` | Next-step LSTM forecast (all modules) |
| `GET` | `/predict/{module}` | Next-step forecast for single module |
| `POST` | `/retrain` | Force retrain from latest data |
