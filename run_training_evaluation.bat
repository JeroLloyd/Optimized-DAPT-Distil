@echo off
TITLE Thesis Experiment - Training & Evaluation (Steps 05-10)
COLOR 0B

:: --- CRITICAL FIX: ACTIVATE VIRTUAL ENVIRONMENT ---
IF EXIST ".venv\Scripts\activate.bat" (
    CALL .venv\Scripts\activate.bat
    ECHO [INFO] Virtual Environment Activated.
) ELSE (
    ECHO [WARNING] .venv not found! Running with global Python...
)

ECHO ========================================================
ECHO      STARTING PARTIAL PIPELINE (Steps 07 - 10)
ECHO      Focus: Training, Optimization, & Benchmarking
ECHO ========================================================
ECHO.


:: --- STEP 7: OPTIMIZATION ---
ECHO.
ECHO [STEP 7/10] Stage 3: Creating Model D (ONNX Quantization)...
python src/07_optimize_stage3_model_d.py
IF %ERRORLEVEL% NEQ 0 GOTO ERROR

:: --- STEP 8: EVALUATION ---
ECHO.
ECHO [STEP 8/10] Collecting Final Metrics (Test Set)...
python src/08_evaluate_metrics.py
IF %ERRORLEVEL% NEQ 0 GOTO ERROR

:: --- STEP 9: BENCHMARKING ---
ECHO.
ECHO [STEP 9/10] Running Throughput & Size Benchmarks...
python src/09_benchmark_simulation.py
IF %ERRORLEVEL% NEQ 0 GOTO ERROR

:: --- STEP 10: VISUALIZATION ---
ECHO.
ECHO [STEP 10/10] Generating Thesis Figures (Pareto, Latency, Accuracy)...
python src/10_visualize_results.py
IF %ERRORLEVEL% NEQ 0 GOTO ERROR

:: --- COMPLETION ---
ECHO.
ECHO ========================================================
ECHO      PARTIAL RUN COMPLETE! SUCCESS!
ECHO      1. Metrics: 'reports/metrics/final_metrics.csv'
ECHO      2. Figures: 'reports/figures/'
ECHO.
ECHO      To launch the Interactive Simulation App, run:
ECHO      streamlit run src/11_simulation_app.py
ECHO ========================================================
PAUSE
EXIT

:ERROR
ECHO.
ECHO !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
ECHO      ERROR DETECTED! PIPELINE HALTED.
ECHO      Please check the error message above.
ECHO !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
PAUSE