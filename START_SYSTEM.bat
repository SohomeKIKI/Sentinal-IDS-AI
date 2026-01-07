@echo off
TITLE AI IDS System Launcher
color 0b

echo ===================================================
echo      SENTINEL AI - IDS SYSTEM LAUNCHER
echo ===================================================
echo.

if not exist "backend" (
    echo [ERROR] Backend folder not found! Are you in the right directory?
    pause
    exit
)

echo [1/5] Checking Dependencies...
pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to install Python dependencies. Please check your Python installation.
    pause
    exit
)

echo.
echo [2/5] Checking Model...
if not exist "backend/sac_actor.pth" (
    echo Model not found. Training new SAC Model...
    python backend/generate_data.py
    python backend/feature_engineering.py
    python backend/train_model.py
)

echo.
echo [3/5] Starting IDS Backend (Detector)...
start "IDS Backend" cmd /k "uvicorn backend.ids_service:app --reload --port 8000"

echo.
echo [4/5] Starting Attacker Console Backend...
start "Attacker Backend" cmd /k "uvicorn backend.attacker_service:app --reload --port 8001"

echo.
echo [5/5] Starting Frontend Dashboard...
cd frontend
if not exist "node_modules" (
    echo Installing Node Modules...
    call npm install
)
start "IDS Dashboard" cmd /k "npm run dev"

echo.
echo ===================================================
echo     SYSTEM IS RUNNING!
echo.
echo     [BLUE TEAM] IDS Dashboard:      http://localhost:5173
echo     [RED TEAM]  Attacker Console:   http://localhost:5173/attack
echo.
echo ===================================================
pause
