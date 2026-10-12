@echo off
REM Starts the FABRIC render API and web shell in two windows.
cd /d %~dp0
start "FABRIC API" cmd /k "call conda activate fabric && uvicorn main:app --port 8010"
start "FABRIC WEB" cmd /k "cd web && npm run dev"
timeout /t 6 >nul
start http://localhost:5180
