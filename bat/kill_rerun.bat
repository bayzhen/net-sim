@echo off
rem ---------------------------------------------------------------------------
rem Kill any process listening on the ports rerun uses, plus any leftover
rem rerun.exe / Python processes started by view_spawn / view_web.
rem
rem Useful when a previous run crashed and the ports are still bound.
rem ---------------------------------------------------------------------------
setlocal enabledelayedexpansion

set "PORTS=9090 9091 9876 19090 19091"

for %%P in (%PORTS%) do (
    for /f "tokens=5" %%A in ('netstat -ano ^| findstr "LISTENING" ^| findstr ":%%P "') do (
        echo killing PID %%A on port %%P
        taskkill /PID %%A /F >nul 2>&1
    )
)

rem Also nuke stray rerun.exe (the SDK-bundled native viewer).
taskkill /IM rerun.exe /F >nul 2>&1

endlocal
