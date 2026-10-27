@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run START_WINDOWS.bat once first to install everything.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m pytest
pause
