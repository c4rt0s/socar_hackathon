@echo off
setlocal
title SOCAR Well Analytics Workbench (VFM x PTA)

set "ROOT=%~dp0"
if exist "%ROOT%data\raw\well_tests.csv" (
    set "VFM_DATA_DIR=%ROOT%data\raw"
)

echo ================================================================
echo  SOCAR Well Analytics Workbench (VFM x PTA)
echo  Starting local server at http://127.0.0.1:8765/ ...
echo ================================================================
echo.

cd /d "%ROOT%vfm_project"
where python >nul 2>&1
if %errorlevel%==0 (
    python dashboard\server.py
) else (
    py -3 dashboard\server.py
)

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Server exited with code %errorlevel%.
    echo Tip: Install dependencies with: pip install pandas numpy scipy scikit-learn lightgbm openpyxl pyarrow
    pause
)
endlocal
