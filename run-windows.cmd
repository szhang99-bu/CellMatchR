@echo off
setlocal
set "repo=%~dp0"
set "runtime=%SystemDrive%\CellMatchR"
if defined CELLMATCHR_WINDOWS_RUNTIME set "runtime=%CELLMATCHR_WINDOWS_RUNTIME%"
set "mamba=%runtime%\Library\bin\micromamba.exe"
set "prefix=%runtime%\CellMatchR-win-env"
if not exist "%mamba%" goto :missing
if not exist "%prefix%\python.exe" goto :missing

set "MAMBA_ROOT_PREFIX=%runtime%"
set "QT_API=pyqt5"
set "PYTHONUTF8=1"
set "PYTHONNOUSERSITE=1"
set "PYTHONPATH=%repo%."
if defined CELLMATCHR_WINDOWS_STARDIST set "PYTHONPATH=%PYTHONPATH%;%CELLMATCHR_WINDOWS_STARDIST%"
if not defined MICROSAM_CACHEDIR set "MICROSAM_CACHEDIR=%runtime%\model-cache\micro-sam"
if not defined CELLPOSE_LOCAL_MODELS_PATH set "CELLPOSE_LOCAL_MODELS_PATH=%runtime%\model-cache\cellpose"

pushd "%repo%" || exit /b 1
"%mamba%" --no-rc run --prefix "%prefix%" python -u "%repo%scripts\run_cellmatchr.py" %*
set "result=%errorlevel%"
popd
if not "%result%"=="0" (
    echo.
    echo CellMatchR exited with code %result%.
    pause
)
exit /b %result%

:missing
echo Run setup-windows.cmd before opening CellMatchR.
pause
exit /b 1
