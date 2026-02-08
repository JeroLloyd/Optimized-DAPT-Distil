@echo off
TITLE Thesis Experiment Pipeline - Automation
COLOR 0A

:: --- CRITICAL FIX: ACTIVATE VIRTUAL ENVIRONMENT ---
:: This ensures Python can find 'datasets', 'torch', etc.
IF EXIST ".venv\Scripts\activate.bat" (
    CALL .venv\Scripts\activate.bat
    ECHO [INFO] Virtual Environment Activated.
) ELSE (
    ECHO [WARNING] .venv not found! Running with global Python...
)

ECHO ========================================================
ECHO      STARTING THESIS EXPERIMENTAL PIPELINE
ECHO      Models: DistilBERT, DAPT-DistilBERT, XLM-R
ECHO ========================================================
ECHO.

:: --- STEP 1: DATA INGESTION ---
ECHO [STEP 1/9] Downloading Datasets...
python src/01_data_ingestion.py
IF %ERRORLEVEL% NEQ 0 GOTO ERROR

:: --- STEP 2: DATA CLEANING ---
ECHO.
ECHO [STEP 2/9] Cleaning Corpora (Gibberish Filtering)...
python src/02_data_cleaning_dapt.py
IF %ERRORLEVEL% NEQ 0 GOTO ERROR

:: --- STEP 3: STRATIFICATION ---
ECHO.
ECHO [STEP 3/9] Stratifying FiReCS (80/10/10 Split)...
python src/03_data_stratification.py
IF %ERRORLEVEL% NEQ 0 GOTO ERROR

:: --- STEP 4: DAPT TRAINING ---
ECHO.
ECHO [STEP 4/9] Stage 1: Domain-Adaptive Pre-Training (Model B Base)...
ECHO (This may take 30-60 minutes depending on your GPU)
python src/04_train_stage1_dapt.py
IF %ERRORLEVEL% NEQ 0 GOTO ERROR

:: --- STEP 5: FINE-TUNING ---
ECHO.
ECHO [STEP 5/9] Stage 2: Supervised Fine-Tuning (Models A, B, C)...
ECHO (Training 3 separate models sequentially...)
python src/05_train_stage2_finetune.py
IF %ERRORLEVEL% NEQ 0 GOTO ERROR

:: --- STEP 6: OPTIMIZATION ---
ECHO.
ECHO [STEP 6/9] Stage 3: Optimization (Model D - INT8/ONNX)...
python src/06_optimize_stage3_model_d.py
IF %ERRORLEVEL% NEQ 0 GOTO ERROR

:: --- STEP 7: EVALUATION ---
ECHO.
ECHO [STEP 7/9] Collecting Final Metrics (Test Set)...
python src/07_evaluate_metrics.py
IF %ERRORLEVEL% NEQ 0 GOTO ERROR

:: --- STEP 8: SIMULATION BENCHMARK ---
ECHO.
ECHO [STEP 8/9] Running Latency Simulation Platform...
python src/08_benchmark_simulation.py
IF %ERRORLEVEL% NEQ 0 GOTO ERROR

:: --- STEP 9: VISUALIZATION ---
ECHO.
ECHO [STEP 9/9] Generating Chapter 4 Figures...
python src/09_visualize_results.py
IF %ERRORLEVEL% NEQ 0 GOTO ERROR

:: --- COMPLETION ---
ECHO.
ECHO ========================================================
ECHO      EXPERIMENT COMPLETE! SUCCESS!
ECHO      Check 'results/metrics_summary.csv' for data.
ECHO      Check 'results/figures/' for your graphs.
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