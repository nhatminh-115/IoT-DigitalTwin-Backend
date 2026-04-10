# Running Dashboard + API Together

## 1) Start Streamlit dashboard

```powershell
python3.11.exe -m streamlit run app.py --server.port 8503
```

## 2) Start FastAPI inference service (separate terminal)

```powershell
python3.11.exe -m uvicorn api_server:app --host 0.0.0.0 --port 8000
```

## 3) Useful API endpoints

- Health check:
  - `GET http://localhost:8000/health`
- Full latest prediction (optional module filter):
  - `GET http://localhost:8000/predict/latest`
  - `GET http://localhost:8000/predict/latest?module=M9`
- Unity-friendly payload:
  - `GET http://localhost:8000/unity/predict?module=M9`
- Force model retraining:
  - `POST http://localhost:8000/retrain`

## 4) Unity polling recommendation (every 180 seconds)

Use `UnityWebRequest.Get("http://<host>:8000/unity/predict?module=M1")` in a 3-minute loop.

## Notes

- API and dashboard share the same checkpoint at `artifacts/lstm_checkpoint.pt`.
- If checkpoint is incompatible, the service will retrain automatically.
