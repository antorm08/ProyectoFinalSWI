@echo off
REM Inicia la API de CommerceAI en http://localhost:8000 (docs en /docs)
cd /d "%~dp0backend"
if not exist ".venv\Scripts\python.exe" (
    echo Creando backend\.venv con Python 3.12...
    py -3.12 -m venv .venv
    if errorlevel 1 (
        echo No fue posible crear el entorno virtual.
        pause
        exit /b 1
    )
)
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m uvicorn app.main:app --reload
