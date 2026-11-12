@echo off
title CineMind AI - Cinema Knowledge Assistant
echo ====================================================================
echo                   🎬  CineMind AI Studio  🎬
echo ====================================================================
echo Starting Streamlit via project virtual environment (.venv)...
cd /d "%~dp0"

if exist ".venv\Scripts\streamlit.exe" (
    ".venv\Scripts\streamlit.exe" run app.py
) else (
    echo [ERROR] Virtual environment not found at .venv!
    echo Please ensure Python 3.11 virtual environment is created.
    pause
)
