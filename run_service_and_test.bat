@echo off
chcp 65001 >nul
echo ============================================================
echo BGE Embedded Service - Start and Test Script
echo ============================================================
echo.

REM Check virtual environment
if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found, please create it first
    pause
    exit /b 1
)

REM Activate virtual environment
echo [1/4] Activating virtual environment...
call .venv\Scripts\activate.bat

REM Check if service is running
echo [2/4] Checking service status...
curl -s http://127.0.0.1:7860/health >nul 2>&1
if %errorlevel% equ 0 (
    echo [INFO] Service is already running
    goto run_tests
)

REM Start service
echo [INFO] Service not running, starting now...
echo [3/4] Starting BGE embedded service...
start "BGE Service" cmd /k ".venv\Scripts\activate.bat && python cherry_bge_service.py"

REM Wait for service to start
echo [INFO] Waiting for service to start...
timeout /t 10 /nobreak >nul

REM Check service status again
curl -s http://127.0.0.1:7860/health >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Service start failed
    pause
    exit /b 1
)

echo [SUCCESS] Service started successfully

:run_tests
echo.
echo [4/4] Running tests...
echo.

REM Run test script
python test_bge_service.py

echo.
echo ============================================================
echo Test Completed
echo ============================================================
pause