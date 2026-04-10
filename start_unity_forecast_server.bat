@echo off
setlocal
cd /d "%~dp0"
if not exist "api_server.py" exit /b 1
if not exist ".venv\Scripts\python.exe" (
  python -m venv .venv
)
set NEED_INSTALL=0
if not exist ".unity_forecast_deps.ok" set NEED_INSTALL=1
".venv\Scripts\python.exe" -c "import pandas,requests,torch,fastapi,uvicorn" >nul 2>&1
if errorlevel 1 set NEED_INSTALL=1
if "%NEED_INSTALL%"=="1" (
  ".venv\Scripts\python.exe" -m pip install --upgrade pip
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt
  if %errorlevel%==0 type nul > ".unity_forecast_deps.ok"
)
".venv\Scripts\python.exe" -m uvicorn api_server:app --host 127.0.0.1 --port 8000
