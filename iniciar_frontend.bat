@echo off
REM Inicia la web de CommerceAI en http://localhost:5173
cd /d "%~dp0frontend"
if not exist "node_modules\.bin\vite.cmd" (
    echo Instalando dependencias del frontend...
    call npm install
    if errorlevel 1 (
        pause
        exit /b 1
    )
)
call npm run dev
if errorlevel 1 (
    pause
)
