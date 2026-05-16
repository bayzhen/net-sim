@echo off
rem ---------------------------------------------------------------------------
rem Serve the dataset over the rerun WEB viewer (browser-based).
rem
rem Note: on Windows the native window (view_spawn.bat) is usually a better
rem experience. Use this only if you really need to share the viewer over
rem the network or run headless.
rem
rem Args:
rem   %1 = raw sample directory   (default: tmp_rerun_10\raw)
rem   %2 = web port               (default: 19090; gRPC port is web+1)
rem
rem Examples:
rem   view_web.bat
rem   view_web.bat tmp_rerun_50\raw 9090
rem ---------------------------------------------------------------------------
setlocal
call "%~dp0_env.bat" || exit /b 1
cd /d "%REPO_DIR%"

set "RAW=%~1"
if "%RAW%"=="" set "RAW=tmp_rerun_10\raw"

set "WEB_PORT=%~2"
if "%WEB_PORT%"=="" set "WEB_PORT=19090"

if not exist "%RAW%" (
    echo [ERROR] raw sample directory not found: %RAW%
    echo Run generate.bat first, or pass the path as argument.
    exit /b 1
)

echo [view-web] loading samples from %RAW%, binding 0.0.0.0:%WEB_PORT%
"%PY%" -m cli view-rerun "%RAW%" --serve --bind 0.0.0.0:%WEB_PORT%

endlocal
