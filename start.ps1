$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $root

Write-Host "Starting StreetPulse backend and frontend..."

Start-Process powershell -ArgumentList @(
    '-NoExit',
    '-Command', "& '$root\\.venv\\Scripts\\Activate.ps1'; Set-Location '$root'; python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000"
)

Start-Process powershell -ArgumentList @(
    '-NoExit',
    '-Command', "Set-Location '$root\\frontend'; npm run dev"
)

Write-Host "Backend will be available at http://127.0.0.1:8000"
Write-Host "Frontend will be available at http://localhost:5173"
Pop-Location
