@echo off
setlocal
cd /d "%~dp0"

set "APP_URL=http://127.0.0.1:5000"
set "LOG_FILE=%~dp0public_tunnel.log"
set "CLOUDFLARED=C:\Program Files (x86)\cloudflared\cloudflared.exe"

echo.
echo Checking Python packages...
python -c "import flask, cv2, mediapipe, joblib, PIL, cryptography" >nul 2>nul
if errorlevel 1 (
  echo Installing required packages from requirements_web.txt...
  python -m pip install -r requirements_web.txt
  if errorlevel 1 (
    echo.
    echo ERROR: Package installation failed.
    echo Check your internet connection, then run this file again.
    echo.
    pause
    exit /b 1
  )
)

where cloudflared >nul 2>nul
if not errorlevel 1 (
  set "CLOUDFLARED=cloudflared"
)

if not exist "%CLOUDFLARED%" (
  if /i not "%CLOUDFLARED%"=="cloudflared" (
    echo.
    echo ERROR: cloudflared was not found.
    echo Install Cloudflare Tunnel or add cloudflared.exe to PATH.
    echo.
    pause
    exit /b 1
  )
)

echo.
echo Starting Sign Language Prediction app...
echo Local URL: %APP_URL%
echo.
start "Sign Language App Server" /min cmd /c "python web_app.py --http-only"

echo Waiting for the local app to start...
timeout /t 4 /nobreak > nul

echo.
echo Creating Cloudflare public HTTPS link...
echo Copy the https://*.trycloudflare.com link shown below.
echo Keep this window open while using the public link.
echo Tunnel log: %LOG_FILE%
echo.

"%CLOUDFLARED%" tunnel --protocol http2 --url %APP_URL% --logfile "%LOG_FILE%" --loglevel info

echo.
echo Public tunnel stopped.
pause
