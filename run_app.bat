@echo off
chcp 65001 > nul
title Blood Cell Detection & Classification XAI System - 1-Click Launcher
cls

echo ===============================================================================
echo   HE THONG PHAT HIEN & PHAN LOAI TE BAO MAU KET HOP XAI (GRADIO WEB DEMO)
echo   De tai Chuyen de CNTT - DH Su pham Ky thuat Vinh Long
echo ===============================================================================
echo.

echo [1/3] Kiem tra moi truong Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [LOI] Khong tim thay Python! Vui long cai dat Python 3.10+ va them vao PATH.
    pause
    exit /b 1
)

echo [2/3] Kiem tra va cai dat thu vien phu thuoc (requirements.txt)...
python -m pip install -q -r requirements.txt

echo [3/3] Dang khoi chay Gradio Web UI...
echo App se tu dong mo tren trinh duyet tai: http://127.0.0.1:7860
echo ===============================================================================
echo.

python notebooks/03_gradio_web_demo.py

pause
