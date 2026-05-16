@echo off
rem ---------------------------------------------------------------------------
rem Open the dataset in a NATIVE rerun viewer window (recommended on Windows).
rem
rem Streams from raw/sample_*.json through an in-process gRPC server, so it
rem ignores any stale dataset.rrd file in the directory.
rem
rem Args:
rem   %1 = directory containing raw\sample_*.json   (default: tmp_rerun_10\raw)
rem
rem Examples:
rem   view_spawn.bat
rem   view_spawn.bat tmp_rerun_50\raw
rem ---------------------------------------------------------------------------
setlocal
call "%~dp0_env.bat" || exit /b 1
cd /d "%REPO_DIR%"

set "RAW=%~1"
if "%RAW%"=="" set "RAW=tmp_rerun_10\raw"

if not exist "%RAW%" (
    echo [ERROR] raw sample directory not found: %RAW%
    echo Run generate.bat first, or pass the path as argument.
    exit /b 1
)

echo [view-spawn] loading samples from %RAW%
"%PY%" -m cli view-rerun "%RAW%" --spawn

endlocal
