@echo off
REM VFX Pipeline Windows Runner
REM Simple batch script to run the VFX pipeline on Windows

echo 🎬 VFX Shot Complexity Prediction Pipeline
echo ==================================================

if "%1"=="" (
    echo Usage: run.bat [local^|docker^|api^|test^|status]
    echo.
    echo Options:
    echo   local    - Run locally with Python
    echo   docker   - Run with Docker Compose
    echo   api      - Run API server only
    echo   test     - Run test suite
    echo   status   - Show system status
    echo.
    goto :end
)

if "%1"=="local" goto :local
if "%1"=="docker" goto :docker
if "%1"=="api" goto :api
if "%1"=="test" goto :test
if "%1"=="status" goto :status

echo ❌ Unknown option: %1
goto :end

:local
echo 🚀 Starting VFX Pipeline (Local Mode)
echo.
echo 🔧 Setting up environment...
python -m venv venv
call venv\Scripts\activate.bat
pip install -r requirements.txt

echo 🚀 Starting services...
docker-compose up -d mongodb redis

echo ⏳ Waiting for services...
timeout /t 10 /nobreak >nul

echo 🎯 Starting pipeline...
echo Drop videos in 'input_files' directory
echo API available at: http://localhost:8000
echo.
python main.py
goto :end

:docker
echo 🐳 Starting VFX Pipeline (Docker Mode)
echo.
echo 🔨 Building and starting...
docker-compose up --build
goto :end

:api
echo 🌐 Starting API Server Only
echo.
echo 🔧 Setting up environment...
python -m venv venv
call venv\Scripts\activate.bat
pip install -r requirements.txt

echo 🚀 Starting services...
docker-compose up -d mongodb redis

echo ⏳ Waiting for services...
timeout /t 10 /nobreak >nul

echo 🎯 Starting API server...
echo Available at: http://localhost:8000
echo Documentation: http://localhost:8000/docs
echo.
uvicorn api.main:app --reload --port 8000
goto :end

:test
echo 🧪 Running Tests
echo.
echo 🔧 Setting up environment...
call venv\Scripts\activate.bat
pip install -r requirements.txt

echo 🚀 Starting services...
docker-compose up -d mongodb redis

echo ⏳ Waiting for services...
timeout /t 10 /nobreak >nul

echo 🧪 Running tests...
pytest -v
goto :end

:status
echo 📊 VFX Pipeline Status
echo ==================================================
echo.
echo 🔧 Services:
docker-compose ps
echo.
echo 🌐 API Health:
curl -s http://localhost:8000/health 2>nul || echo API not available
echo.
echo 💾 Input Files:
dir input_files 2>nul || echo No input files directory
goto :end

:end
echo.
echo 🎬 Done!
pause
