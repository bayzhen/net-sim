@echo off
rem ---------------------------------------------------------------------------
rem Save the dataset to a single .rrd file you can later open with the
rem standalone rerun viewer (no Python needed):
rem     rerun.exe path\to\dataset.rrd
rem
rem Args:
rem   %1 = raw sample directory   (default: tmp_rerun_10\raw)
rem   %2 = output .rrd path       (default: <raw_parent>\dataset.rrd)
rem
rem Examples:
rem   view_save_rrd.bat
rem   view_save_rrd.bat tmp_rerun_50\raw shared\run50.rrd
rem ---------------------------------------------------------------------------
setlocal
call "%~dp0_env.bat" || exit /b 1
cd /d "%REPO_DIR%"

set "RAW=%~1"
if "%RAW%"=="" set "RAW=tmp_rerun_10\raw"

set "OUT_RRD=%~2"
if "%OUT_RRD%"=="" (
    rem default: sibling of raw\, named dataset.rrd
    for %%I in ("%RAW%\..") do set "OUT_RRD=%%~fI\dataset.rrd"
)

if not exist "%RAW%" (
    echo [ERROR] raw sample directory not found: %RAW%
    exit /b 1
)

echo [save-rrd] %RAW%  ->  %OUT_RRD%
"%PY%" -m cli view-rerun "%RAW%" --save "%OUT_RRD%"

endlocal
