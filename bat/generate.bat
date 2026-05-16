@echo off
rem ---------------------------------------------------------------------------
rem Generate dataset samples (Warp + CUDA, with raw frames included).
rem
rem Args (all optional, in order):
rem   %1 = output directory      (default: tmp_rerun_10)
rem   %2 = number of samples     (default: 10)
rem   %3 = seed                  (default: 1)
rem
rem Examples:
rem   generate.bat
rem   generate.bat tmp_rerun_50 50
rem   generate.bat my_runs\run01 100 42
rem ---------------------------------------------------------------------------
setlocal
call "%~dp0_env.bat" || exit /b 1
cd /d "%REPO_DIR%"

set "OUT=%~1"
if "%OUT%"=="" set "OUT=tmp_rerun_10"

set "COUNT=%~2"
if "%COUNT%"=="" set "COUNT=10"

set "SEED=%~3"
if "%SEED%"=="" set "SEED=1"

echo [generate] device=cuda count=%COUNT% seed=%SEED% output=%OUT%
"%PY%" -m cli generate ^
    --count %COUNT% ^
    --seed %SEED% ^
    --device cuda ^
    --raw ^
    --output "%OUT%"

endlocal
