---
title: IoT Digital Twin Dashboard
emoji: 🏭
colorFrom: indigo
colorTo: blue
sdk: streamlit
sdk_version: "1.39.0"
python_version: "3.11"
app_file: app.py
pinned: false
---

# IoT Digital Twin Backend & Forecast Dashboard

This repository contains the backend service and real-time forecasting dashboard for the IoT Digital Twin project. 

## Features
- **Real-time Monitoring**: Streamlit-based dashboard for sensor data visualization.
- **Multivariate Forecasting**: LSTM and Autoencoder models for predictive analytics.
- **REST API**: FastAPI endpoints for Unity integration and external data access.
- **Automated Sync**: Synchronized via GitHub Actions to Hugging Face Spaces.

## Getting Started
To run locally:
1. `pip install -r requirements.txt`
2. `streamlit run app.py` (Dashboard)
3. `uvicorn api_server:app --reload` (API Server)

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
