# Architecture documentation — IoT Digital Twin Backend

This directory documents the **as-implemented** architecture of this repository, reviewed against the default branch source on **2026-10-19**. It is a code-grounded engineering reference, not a proposed future system design.

## Reading order

1. [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md) — deployed processes, ownership boundaries, component map, storage, and deployment topology.
2. [DATA_FLOWS.md](DATA_FLOWS.md) — end-to-end telemetry, quality control, prediction, Unity, Telegram/AI, MQTT, weather, visualization, and command flow; includes Mermaid sequence diagrams.
3. [API_AND_CONTRACTS.md](API_AND_CONTRACTS.md) — actual HTTP routes, canonical schemas, endpoint response shapes, sample payloads, timing and errors.
4. [OPERATIONS_AND_CAVEATS.md](OPERATIONS_AND_CAVEATS.md) — configuration, startup, health, runtime failure scenarios, and code-level limitations.

## The key architectural distinction

**Primary live inference path:** Home Assistant → Google Sheets CSV → `DataFetcher` → `DataQualityGate` → local model/cache → FastAPI, Unity, Streamlit, Telegram. Home Assistant → Sheets is described in the existing project [README](../README.md), but the integration running that upstream publication is **not implemented in this repository**.

**Independent optional ingestion path:** ESP MQTT devices → HiveMQ Cloud (`esp/+`) → `mqtt_worker.py` → Supabase `env_readings`. The inference service does **not** read `env_readings` to make forecasts. Other optional Supabase tables log alerts, bot interactions, and heartbeats.

Source entry points: [`api_server.py`](../api_server.py), [`src/iot_digital_twin/api_service.py`](../src/iot_digital_twin/api_service.py), [`app.py`](../app.py), [`mqtt_worker.py`](../mqtt_worker.py).

## Interpretation and provenance

- **Implemented** means the described behavior is visible in code or process configuration.
- **Documented upstream** means the repository README describes an external feed but the producer/integration cannot be verified from this repository alone.
- **Optional/conditional** means a component only works with configuration, checkpoints, credentials, or assets.
- **Not demonstrated** means this repository does not show the entire interaction, even if an external project might implement it.

The documents describe code behavior; they do not assert that a cloud deployment, external sensor network, Telegram bot, model weights, or Supabase database is currently healthy. No credentials or private payloads are included. Times and sample values are illustrative unless explicitly identified as configuration constants.

## Quick navigation

| Question | Go to |
|---|---|
| What runs in the Docker container? | [Deployment topology](SYSTEM_ARCHITECTURE.md#deployment-topology) |
| Which service actually gets fresh telemetry? | [Primary inference path](DATA_FLOWS.md#1-primary-telemetry-and-inference-path) |
| Is MQTT the model input? | [Independent MQTT path](DATA_FLOWS.md#5-independent-mqtt-to-supabase-ingestion) |
| How does Unity call predictions? | [Unity contract](API_AND_CONTRACTS.md#unity-prediction-contract) |
| How does Telegram/Groq get its context? | [Bot and AI flow](DATA_FLOWS.md#4-telegram-command-ai-and-alert-paths) |
| What could break in multi-worker deployment? | [Known caveats](OPERATIONS_AND_CAVEATS.md#known-implementation-caveats) |
