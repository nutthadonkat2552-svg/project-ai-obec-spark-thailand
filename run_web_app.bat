@echo off
cd /d "%~dp0"

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

echo.
echo Open on this computer:
echo http://127.0.0.1:5000/
echo.
echo Open on your phone with the computer IP shown below.
echo Use https://COMPUTER-IP:5443/ and accept the certificate warning once.
echo If Windows asks about firewall access, choose Allow.
echo.
python web_app.py
pause
