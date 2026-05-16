@echo off
rem ---------------------------------------------------------------------------
rem Shared environment for all goal-net-xpbd bat scripts.
rem
rem Sets:
rem   PY        - python interpreter to use (must have rerun-sdk, warp-lang,
rem               numpy installed; see `pip list` to verify)
rem   REPO_DIR  - absolute path of the repo root (parent of this bat folder)
rem
rem Override PY by exporting it before calling any script, e.g.
rem   set PY=C:\path\to\other\python.exe & call run_view_spawn.bat
rem ---------------------------------------------------------------------------

if not defined PY set "PY=C:\Python312\python.exe"

rem REPO_DIR = parent of this _env.bat's directory.
for %%I in ("%~dp0..") do set "REPO_DIR=%%~fI"

if not exist "%PY%" (
    echo [ERROR] Python interpreter not found: %PY%
    echo Set PY env var to a Python 3.10+ with rerun-sdk/warp-lang installed.
    exit /b 1
)
