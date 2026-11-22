@echo off
REM Inicia la API de CommerceAI en http://localhost:8000 (docs en /docs)
cd /d "%~dp0backend"
call .venv\Scripts\activate.bat
uvicorn app.main:app --reload
