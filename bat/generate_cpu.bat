@echo off
rem ---------------------------------------------------------------------------
rem Same as generate.bat but uses CPU. Useful when CUDA / Warp kernel cache
rem misbehaves, or on machines without an NVIDIA GPU.
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

echo [generate-cpu] device=cpu count=%COUNT% seed=%SEED% output=%OUT%
"%PY%" -m cli generate ^
    --count %COUNT% ^
    --seed %SEED% ^
    --device cpu ^
    --raw ^
    --output "%OUT%"

endlocal
