@echo off
rem ---------------------------------------------------------------------------
rem One-shot helper: kill anything on rerun ports, regenerate a fresh dataset,
rem then open the native viewer window.
rem
rem Args (all optional):
rem   %1 = output dir   (default: tmp_rerun_10)
rem   %2 = sample count (default: 10)
rem   %3 = seed         (default: 1)
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

echo [step 1/3] killing stale rerun processes (ports 9090/9091/9876/19090/19091)
call "%~dp0kill_rerun.bat" >nul

echo [step 2/3] generating %COUNT% samples into %OUT%
call "%~dp0generate.bat" "%OUT%" %COUNT% %SEED% || (
    echo [ERROR] generate failed
    exit /b 1
)

echo [step 3/3] opening native rerun viewer
call "%~dp0view_spawn.bat" "%OUT%\raw"

endlocal
