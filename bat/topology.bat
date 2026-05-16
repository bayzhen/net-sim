@echo off
rem ---------------------------------------------------------------------------
rem Print topology summary for the default GoalNetParams.
rem Optional first arg: path to params JSON.
rem
rem Usage:
rem   topology.bat
rem   topology.bat my_params.json
rem ---------------------------------------------------------------------------
setlocal
call "%~dp0_env.bat" || exit /b 1
cd /d "%REPO_DIR%"

if "%~1"=="" (
    "%PY%" -m cli topology
) else (
    "%PY%" -m cli --params "%~1" topology
)

endlocal
