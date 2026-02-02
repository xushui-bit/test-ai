@echo off
chcp 65001 >nul
echo ============================================================
echo BGE Embedded Service - Quick Start
echo ============================================================
echo.

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Start service
python cherry_bge_service.py

pause