@echo off
title FreightSense Launcher
echo ===================================================
echo   🚛 FreightSense AI Logistics Engine Launcher 🚛
echo ===================================================
echo.

:: Check if virtual environment exists
if not exist "venv" (
    echo [ERROR] Virtual environment 'venv' not found in the current directory!
    echo Please ensure the venv folder exists before running this launcher.
    echo.
    pause
    exit /b
)

echo 📦 Installing / Updating dependencies...
.\venv\Scripts\python -m pip install --upgrade pip setuptools wheel -q
.\venv\Scripts\pip install -r requirements.txt -q

echo 🤖 Starting FastAPI Backend (Port 8000)...
start "FreightSense Backend" cmd /k "echo 🚀 Starting FastAPI Backend API... && .\venv\Scripts\uvicorn src.api.main:app --host 0.0.0.0 --port 8000"

:: Give the backend 2 seconds to bind to port 8000
timeout /t 2 /nobreak > nul

echo 🎨 Starting Streamlit Frontend Dashboard (Port 8501)...
start "FreightSense Frontend Dashboard" cmd /k "echo 🚀 Starting Streamlit App... && .\venv\Scripts\streamlit run app.py"

echo.
echo ===================================================
echo   ✅ Both servers have been successfully dispatched!
echo   - Backend API: http://localhost:8000
echo   - Streamlit UI: http://localhost:8501
echo ===================================================
echo.
pause
