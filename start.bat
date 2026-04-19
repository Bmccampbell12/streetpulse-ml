@echo off
set ROOT=%~dp0
cd /d "%ROOT%"

echo Starting StreetPulse backend and frontend...
start "StreetPulse Backend" powershell -NoExit -Command "& '%ROOT%\.venv\Scripts\Activate.ps1'; Set-Location '%ROOT%'; python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000"
start "StreetPulse Frontend" powershell -NoExit -Command "Set-Location '%ROOT%frontend'; npm run dev"

echo Backend will be available at http://127.0.0.1:8000
echo Frontend will be available at http://localhost:5173
pause
