@echo off
chcp 65001 >nul
cd /d %~dp0..
echo ============================================================
echo BGE Embedded Service - Cache Management API Test Script
echo ============================================================
echo.

REM Activate virtual environment
echo [INFO] Activating virtual environment...
call .venv\Scripts\activate.bat

echo.
echo ============================================================
echo Testing Cache Management API
echo ============================================================
echo.

REM 1. Get cache statistics
echo [1/6] Getting cache statistics...
curl -s http://127.0.0.1:7860/cache/stats
echo.
echo.

REM 2. Test cache effect (first request)
echo [2/6] Testing cache effect - First request (no cache)...
curl -s -X POST http://127.0.0.1:7860/embed -H "Content-Type: application/json" -d "{\"texts\": [\"Test cache effect text\"], \"normalize\": true}"
echo.
echo.

REM 3. Test cache effect (second request)
echo [3/6] Testing cache effect - Second request (with cache)...
curl -s -X POST http://127.0.0.1:7860/embed -H "Content-Type: application/json" -d "{\"texts\": [\"Test cache effect text\"], \"normalize\": true}"
echo.
echo.

REM 4. Get cache statistics again
echo [4/6] Getting cache statistics again...
curl -s http://127.0.0.1:7860/cache/stats
echo.
echo.

REM 5. Manually save cache
echo [5/6] Manually saving cache to file...
curl -s -X POST http://127.0.0.1:7860/cache/save
echo.
echo.

REM 6. Remove expired cache
echo [6/6] Removing expired cache entries...
curl -s -X POST http://127.0.0.1:7860/cache/remove-expired
echo.
echo.

echo ============================================================
echo Cache Management API Test Completed
echo ============================================================
echo.
echo Available Cache Management APIs:
echo   GET    /cache/stats      - Get cache statistics
echo   POST   /cache/clear      - Clear cache
echo   POST   /cache/save       - Save cache to file
echo   POST   /cache/remove-expired - Remove expired entries
echo.
echo Example commands:
echo   curl http://127.0.0.1:7860/cache/stats
echo   curl -X POST http://127.0.0.1:7860/cache/clear
echo.
pause
