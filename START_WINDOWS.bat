@echo off
setlocal
title Latent Space Explorer
cd /d "%~dp0"
echo ============================================================
echo   Latent Space Explorer  -  StyleGAN2 latent interpolation
echo ============================================================
echo.

REM ---- find Python (3.9 or newer) ----
set "PY="
where py >nul 2>nul && (py -3 -c "import sys; sys.exit(0 if sys.version_info>=(3,9) else 1)" >nul 2>nul && set "PY=py -3")
if not defined PY (
  where python >nul 2>nul && (python -c "import sys; sys.exit(0 if sys.version_info>=(3,9) else 1)" >nul 2>nul && set "PY=python")
)
if not defined PY (
  echo [ERROR] Python 3.9+ was not found.
  echo Install it from https://www.python.org/downloads/  and TICK "Add python.exe to PATH".
  echo Then double-click START_WINDOWS.bat again.
  pause
  exit /b 1
)
echo Using: %PY%
%PY% --version

REM ---- one-time setup: virtual environment + packages ----
if not exist ".venv\Scripts\python.exe" (
  echo.
  echo [1/3] Creating virtual environment .venv ...
  %PY% -m venv .venv || (echo [ERROR] could not create venv & pause & exit /b 1)
)
set "VPY=.venv\Scripts\python.exe"

if not exist ".venv\.installed" (
  echo.
  echo [2/3] Installing packages - first run only, takes a few minutes ...
  "%VPY%" -m pip install --upgrade pip
  "%VPY%" -m pip install torch --index-url https://download.pytorch.org/whl/cpu || "%VPY%" -m pip install torch
  "%VPY%" -m pip install -r requirements.txt || (echo [ERROR] package install failed & pause & exit /b 1)
  echo done> ".venv\.installed"
)

REM ---- one-time: download pretrained weights (cached in weights\) ----
echo.
echo [3/3] Checking pretrained StyleGAN2 weights ...
"%VPY%" download_model.py --no-verify
if errorlevel 1 (
  echo.
  echo [WARNING] Weights could not be downloaded. See the link above, or README - Troubleshooting.
  pause
)

if not exist "Latent_Space_Explorer_OFFLINE.html" if exist "weights\*.pkl" (
  echo.
  echo [extra] Building Latent_Space_Explorer_OFFLINE.html - one-time, about 1-3 minutes ...
  echo         That single file opens on ANY laptop by double-click - no Python needed.
  "%VPY%" make_offline_page.py
)

echo.
echo Starting the app - your browser will open at http://127.0.0.1:8765
echo Keep this window open while using the app. Close it (or press Ctrl+C) to stop.
echo.
"%VPY%" app.py %*
pause
