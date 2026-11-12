# CineMind AI - PowerShell Launcher
$ScriptDir = Split-Path -Parent -Path $MyInvocation.MyCommand.Definition
Set-Location $ScriptDir

Write-Host "====================================================================" -ForegroundColor Cyan
Write-Host "                  🎬  CineMind AI Studio  🎬" -ForegroundColor Yellow
Write-Host "====================================================================" -ForegroundColor Cyan
Write-Host "Launching CineMind AI using project virtual environment (.venv)..." -ForegroundColor Green

if (Test-Path ".\.venv\Scripts\streamlit.exe") {
    & ".\.venv\Scripts\streamlit.exe" run app.py
} else {
    Write-Host "[ERROR] Virtual environment not found at .venv!" -ForegroundColor Red
    pause
}
