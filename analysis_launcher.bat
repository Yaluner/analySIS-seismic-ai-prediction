@echo off
TITLE Deprem Sistemi Baslatici
color 0A

:: 1. SET LOCATION
cd /d "%~dp0"

:: 2. CHECK PYTHON
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [HATA] Python bulunamadi.
    pause
    exit
)

:: 3. LAUNCH
echo.
echo [BILGI] Sistem baslatiliyor...
echo.
echo ------------------------------------------------
echo Eger tarayici acilmazsa, lutfen su adrese gidin:
echo http://localhost:8501
echo ------------------------------------------------
echo.

:: (Waiting 2 seconds for server to warm up)
timeout /t 2 >nul

:: Run Streamlit (Without the "headless" restriction)
python -m streamlit run app.py

pause