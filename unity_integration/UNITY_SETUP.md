# Unity Integration (Object/UI)

## 1) Copy scripts into Unity project

Copy these two files into your Unity `Assets/Scripts` folder:

- `unity_integration/UnityInferenceClient.cs`
- `unity_integration/UnityInferenceUIBinder.cs`

Both scripts use namespace `DigitalTwin.Inference`.

## 2) Scene setup

1. Create an empty GameObject named `InferenceClient`.
2. Attach `UnityInferenceClient`.
3. Set:
   - `Api Base Url`: `http://<python-host-ip>:8000`
   - `Module Code`: `M1` (or `M4`, `M6`, `M7`, `M8`, `M9`, `M10`, `M11`)
   - `Poll Interval Seconds`: `180`
4. Create another GameObject for UI binder and attach `UnityInferenceUIBinder`.
5. Assign TextMeshPro UI references:
   - `Temp Text`, `Humid Text`, `CO2 Text`, `TVOC Text`
   - `Module Text`, `Issue Time Text`, `Target Time Text`, `Status Text`
6. Drag `InferenceClient` into `inferenceClient` field on binder.

## 3) Python backend

Run API service from this project:

```powershell
python3.11.exe -m uvicorn api_server:app --host 0.0.0.0 --port 8000
```

## 4) Endpoint used by Unity

- `GET /unity/predict?module=M1`

Response now includes `features` array for direct `JsonUtility` parsing:

```json
{
  "module": "M1",
  "forecast_issue_time": "...",
  "forecast_target_time": "...",
  "features": [
    {"name": "Temp", "current": 26.1, "predicted": 26.11, "delta": 0.01},
    {"name": "Humid", "current": 77.1, "predicted": 77.08, "delta": -0.02}
  ]
}
```

## Notes

- Keep API and Unity clocks reasonably synced for easier debugging.
- If Unity runs on another machine, allow inbound TCP `8000` on backend firewall.
