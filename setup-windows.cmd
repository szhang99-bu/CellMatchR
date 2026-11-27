@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\setup_windows.ps1" %*
if errorlevel 1 (
    echo.
    echo CellMatchR setup failed. See the error and setup log path above.
    pause
    exit /b 1
)
pause
