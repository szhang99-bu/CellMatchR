@echo off
setlocal
cd /d "%~dp0"

set "QT_API=pyqt5"
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"
set "PYTHONNOUSERSITE=1"

set "APP_PYTHON="
if defined CELLMATCHR_WINDOWS_PYTHON if exist "%CELLMATCHR_WINDOWS_PYTHON%" set "APP_PYTHON=%CELLMATCHR_WINDOWS_PYTHON%"
if not defined APP_PYTHON if /I "%CONDA_DEFAULT_ENV%"=="cellmatchr-windows" if exist "%CONDA_PREFIX%\python.exe" set "APP_PYTHON=%CONDA_PREFIX%\python.exe"
if not defined APP_PYTHON for %%B in ("%USERPROFILE%\miniforge3" "%LOCALAPPDATA%\miniforge3" "%PROGRAMDATA%\miniforge3" "%USERPROFILE%\miniconda3" "%USERPROFILE%\anaconda3") do if not defined APP_PYTHON if exist "%%~B\envs\cellmatchr-windows\python.exe" set "APP_PYTHON=%%~B\envs\cellmatchr-windows\python.exe"

if defined APP_PYTHON goto run_direct
where conda >nul 2>nul
if errorlevel 1 goto missing_env
conda run --no-capture-output -n cellmatchr-windows python "%~dp0scripts\run_cellmatchr.py"
set "APP_EXIT=%ERRORLEVEL%"
goto finished

:run_direct
"%APP_PYTHON%" "%~dp0scripts\run_cellmatchr.py"
set "APP_EXIT=%ERRORLEVEL%"
goto finished

:missing_env
echo CellMatchR Windows environment was not found.
echo Open the Miniforge Prompt and follow WINDOWS_PREVIEW.md.
echo For a custom environment path, set CELLMATCHR_WINDOWS_PYTHON to its python.exe.
set "APP_EXIT=1"

:finished

if not "%APP_EXIT%"=="0" (
    echo.
    echo CellMatchR exited with code %APP_EXIT%.
    pause
)
exit /b %APP_EXIT%
