@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion

REM Move to repository root directory
cd /d "%~dp0\.."

echo ==============================================================
echo   Chip Anomaly Detection System - Quick Launcher (Windows)
echo ==============================================================

where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Khong tim thay Python trong PATH!
    echo Vui long cai dat Python 3.10 - 3.12 tu https://www.python.org/downloads/
    echo ^(Nho tich chon "Add Python to PATH" khi cai dat^).
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo [1/3] Dang tao moi truong ao Python ^(.venv^)...
    python -m venv .venv
    if %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Khong the tao moi truong ao .venv.
        pause
        exit /b 1
    )
)

echo [2/3] Kiem tra va cai dat thu vien can thiet...
".venv\Scripts\python.exe" -m pip install --upgrade pip -q
".venv\Scripts\python.exe" -c "import streamlit, torch, torchvision, cv2, scipy, yaml" >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Dang cai dat cac goi trong requirements.txt ^(lan dau co the mat vai phut^)...
    ".venv\Scripts\pip.exe" install -r requirements.txt
    if %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Cai dat thu vien that bai! Kiem tra ket noi mang hoac phien ban Python.
        pause
        exit /b 1
    )
)

if not exist "weights\patchcore_best.ckpt" (
    echo [WARNING] Khong tim thay weights\patchcore_best.ckpt! Kiem tra lai repo da clone day du chua.
)

echo [3/3] Khoi dong giao dien Web UI ^(Streamlit^)...
echo Trinh duyet se tu dong mo tai: http://localhost:8501
".venv\Scripts\streamlit.exe" run src\ui\app.py
pause
