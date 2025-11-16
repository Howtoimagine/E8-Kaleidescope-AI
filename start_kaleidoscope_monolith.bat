@echo off
cd /d "%~dp0"
setlocal ENABLEDELAYEDEXPANSION
chcp 65001 >nul

REM --- Conda detection / configuration ---------------------------------------
if "%E8_CONDA_ENV%"=="" set "E8_CONDA_ENV=kaleidoscope"
if "%E8_USE_CONDA%"=="" set "E8_USE_CONDA=auto"
set "CONDA_BAT="
for /f "delims=" %%I in ('where conda.bat 2^>nul') do (
        if not defined CONDA_BAT set "CONDA_BAT=%%~fI"
)
if not defined CONDA_BAT (
        for /f "delims=" %%I in ('where conda 2^>nul') do (
                if not defined CONDA_BAT set "CONDA_BAT=%%~fI"
        )
)
set "USE_CONDA_RUN=0"
if /I not "%E8_USE_CONDA%"=="0" (
        if defined CONDA_BAT (
                set "USE_CONDA_RUN=1"
        ) else if /I "%E8_USE_CONDA%"=="1" (
                echo [WARN] Requested conda mode but no 'conda' executable was found on PATH.
        )
)

REM Prefer project venv's python if available
if "%USE_CONDA_RUN%"=="1" (
        set "PY=python"
) else (
        if exist .venv\Scripts\python.exe (
                set "PY=.venv\Scripts\python.exe"
        ) else (
                set "PY=python"
        )
)

echo.
echo === E8 Kaleidoscope Monolith - M27.FIELDS Enabled ===
echo This launches the monolith server in INTERACTIVE mode by default.
echo M27 Deep Genius (E8/Leech lattice geometry) + M27.FIELDS (Cognitive Field Dynamics) are ENABLED by default.
echo Self-Projection and Data Ingestion are ENABLED.
echo.

REM --- Runtime defaults (user may override via environment before launching) ---
if "%E8_MAX_STEPS%"=="" set "E8_MAX_STEPS=297000"
if "%MIND_PROFILE%"=="" set "MIND_PROFILE=default"
if "%E8_PROVIDER%"=="" set "E8_PROVIDER=ask"
if "%E8_UI_RAY_ALERTS_DEMO%"=="" set "E8_UI_RAY_ALERTS_DEMO=0"

REM Boundary force defaults
if "%E8_BOUNDARY_FORCE_INTERVAL%"=="" set "E8_BOUNDARY_FORCE_INTERVAL=333"
if "%E8_BOUNDARY_FORCE_MIN_CLUSTER%"=="" set "E8_BOUNDARY_FORCE_MIN_CLUSTER=6"

REM Core loops ON
if "%ENABLE_TEACHER%"=="" set "ENABLE_TEACHER=1"
if "%ENABLE_EXPLORER%"=="" set "ENABLE_EXPLORER=1"
if "%ENABLE_INSIGHT%"=="" set "ENABLE_INSIGHT=1"
if "%ENABLE_DREAM%"=="" set "ENABLE_DREAM=1"

REM Engine mode + checkpoints + budgets
if "%MODE%"=="" set "MODE=QUANTUM"
if "%RUN_DIR%"=="" set "RUN_DIR=./runs/run_S4_full"
if "%STATE_EVERY%"=="" set "STATE_EVERY=200"
if "%VALIDATOR_BUDGET_MS%"=="" set "VALIDATOR_BUDGET_MS=300"

REM Ablations OFF (0 means "do NOT ablate" ? keep the feature enabled)
if "%ABLT_RAY%"=="" set "ABLT_RAY=0"
if "%ABLT_VALIDATORS%"=="" set "ABLT_VALIDATORS=0"
if "%ABLT_DIVERSITY%"=="" set "ABLT_DIVERSITY=0"
if "%ABLT_CURRICULUM%"=="" set "ABLT_CURRICULUM=0"

REM Newly requested threshold and tuning defaults (only set if not already defined)
if "%E8_RAY_ALERT_THRESH%"=="" set "E8_RAY_ALERT_THRESH=0.60"
if "%E8_RAY_LOCK_THRESH%"=="" set "E8_RAY_LOCK_THRESH=0.45"
if "%E8_TEACHER_SMALL_GRAPH_NODES%"=="" set "E8_TEACHER_SMALL_GRAPH_NODES=50"
if "%E8_LLM_MAX_RETRIES%"=="" set "E8_LLM_MAX_RETRIES=2"
if "%E8_LLM_RETRY_BACKOFF%"=="" set "E8_LLM_RETRY_BACKOFF=0.6"
if "%E8_CADENCE_PROFILE%"=="" set "E8_CADENCE_PROFILE=m20"
if "%E8_CADENCE_SCALE%"=="" set "E8_CADENCE_SCALE=1.0"
if "%E8_TEACHER_ASK_EVERY%"=="" set "E8_TEACHER_ASK_EVERY=20"
if "%E8_TEACHER_OFFSET%"=="" set "E8_TEACHER_OFFSET=3"
if "%E8_EXPLORER_OFFSET%"=="" set "E8_EXPLORER_OFFSET=10"
if "%E8_LOG_SCHEDULER%"=="" set "E8_LOG_SCHEDULER=1"
if "%BETA_VAE_WARMUP%"=="" set "BETA_VAE_WARMUP=300"

REM Enable enhanced console telemetry by default
if "%E8_CONSOLE_MODE%"=="" set "E8_CONSOLE_MODE=engine"
if "%E8_CONSOLE_VERBOSITY%"=="" set "E8_CONSOLE_VERBOSITY=2"

REM Create runtime directory if it doesn't exist
if not exist "runtime" mkdir "runtime"

if "%E8_CONSOLE_JSON%"=="" set "E8_CONSOLE_JSON=%CD%\runtime\console.ndjson"

REM Dialogue logging shares the same NDJSON stream
if "%E8_UI_DIALOGUE_JSON%"=="" set "E8_UI_DIALOGUE_JSON=%E8_CONSOLE_JSON%"

REM Enable self-projection and data ingestion by default
if "%E8_SELF_PROJECT%"=="" set "E8_SELF_PROJECT=1"
if "%E8_INGEST%"=="" set "E8_INGEST=1"

REM Ensure UTF-8 runtime for consistent logging
set "PYTHONUTF8=1"

REM Select target script (prefer M25.1, allow override via E8_TARGET_SCRIPT)
set "TARGET="
if not "%E8_TARGET_SCRIPT%"=="" (
        if exist "%E8_TARGET_SCRIPT%" (
                set "TARGET=%E8_TARGET_SCRIPT%"
        ) else (
                echo [WARN] Requested E8_TARGET_SCRIPT "%E8_TARGET_SCRIPT%" was not found, falling back to autodetect.
        )
)

if not defined TARGET (
        if exist e8_mind_server_M25.1.py (
                set "TARGET=e8_mind_server_M25.1.py"
        ) else if exist e8_mind_server_M25.py (
                set "TARGET=e8_mind_server_M25.py"
        ) else if exist e8_mind_server_M24.6.py (
                set "TARGET=e8_mind_server_M24.6.py"
        ) else if exist e8_mind_server_M24.4.py (
                set "TARGET=e8_mind_server_M24.4.py"
        ) else (
                echo ERROR: Could not find e8_mind_server_M25.1.py, e8_mind_server_M25.py, e8_mind_server_M24.6.py, or e8_mind_server_M24.4.py in %CD%
                echo        Make sure you're in the project root.
                exit /b 1
        )
)

REM Derive and display version label from selected target
set "VER_LABEL=unknown"
if "%TARGET%"=="e8_mind_server_M25.1.py" set "VER_LABEL=M25.1"
if "%TARGET%"=="e8_mind_server_M25.py" set "VER_LABEL=M25"
if "%TARGET%"=="e8_mind_server_M24.6.py" set "VER_LABEL=M24.6"
if "%TARGET%"=="e8_mind_server_M24.4.py" set "VER_LABEL=M24.4"
echo Monolith Version: %VER_LABEL%
set "E8_APP_VERSION=%VER_LABEL%"

if "%USE_CONDA_RUN%"=="1" (
        set "PY_DISPLAY=conda (%E8_CONDA_ENV%)"
) else (
        set "PY_DISPLAY=%PY%"
)

echo Environment:
echo   E8_PROVIDER=%E8_PROVIDER%  MIND_PROFILE=%MIND_PROFILE%  E8_MAX_STEPS=%E8_MAX_STEPS%
echo.
if "%USE_CONDA_RUN%"=="1" (
        echo Using Conda environment: %E8_CONDA_ENV%
        echo Conda shim: %CONDA_BAT%
)
echo Launching: %PY_DISPLAY% -u %TARGET%
echo Server URL: http://localhost:7871/
echo (Press Ctrl+C to stop)
echo.

REM --- Launch with fallback and pause-on-error ---------------------------------
set "EXITCODE=0"
if "%USE_CONDA_RUN%"=="1" (
        call "%CONDA_BAT%" run -n "%E8_CONDA_ENV%" python -u "%TARGET%"
        set "EXITCODE=!ERRORLEVEL!"
        if not "!EXITCODE!"=="0" (
                echo [WARN] Conda run failed with code !EXITCODE!. Falling back to local Python...
                if exist .venv\Scripts\python.exe (
                        .venv\Scripts\python.exe -u "%TARGET%"
                ) else (
                        python -u "%TARGET%"
                )
                set "EXITCODE=!ERRORLEVEL!"
        )
) else (
        "%PY%" -u "%TARGET%"
        set "EXITCODE=!ERRORLEVEL!"
)

REM Optionally open the UI in the default browser if the server is likely running locally
REM Uncomment the next line if you want the browser to auto-open:
REM start "" http://localhost:7871/

if not "%EXITCODE%"=="0" (
        echo.
        echo [ERROR] Monolith exited with code %EXITCODE%.
        echo Press any key to close this window...
        pause >nul
)

endlocal
