@echo off
setlocal
cd /d "%~dp0"
if not exist "app.py" exit /b 1
if not exist ".venv\Scripts\python.exe" (
  python -m venv .venv
)
set NEED_INSTALL=0
if not exist ".unity_forecast_deps.ok" set NEED_INSTALL=1
".venv\Scripts\python.exe" -c "import pandas,plotly,streamlit,streamlit_autorefresh,torch" >nul 2>&1
if errorlevel 1 set NEED_INSTALL=1
if "%NEED_INSTALL%"=="1" (
  ".venv\Scripts\python.exe" -m pip install --upgrade pip
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt
  if %errorlevel%==0 type nul > ".unity_forecast_deps.ok"
)
set PORT=8503
if not "%MODEL_DASHBOARD_PORT%"=="" set PORT=%MODEL_DASHBOARD_PORT%
".venv\Scripts\python.exe" -m streamlit run app.py --server.port %PORT% --server.headless true --browser.gatherUsageStats false
