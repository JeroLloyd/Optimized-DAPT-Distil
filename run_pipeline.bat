@echo off
TITLE Thesis Experiment Pipeline - Automation
COLOR 0A

:: --- CRITICAL FIX: ACTIVATE VIRTUAL ENVIRONMENT ---
:: This ensures Python can find 'datasets', 'torch', 'optimum', etc.
IF EXIST ".venv\Scripts\activate.bat" (
    CALL .venv\Scripts\activate.bat
    ECHO [INFO] Virtual Environment Activated.
) ELSE (
    ECHO [WARNING] .venv not found! Running with global Python...
)

ECHO ========================================================
ECHO      STARTING THESIS EXPERIMENTAL PIPELINE
ECHO      Focus: DAPT-DistilBERT & ONNX Optimization
ECHO ========================================================
ECHO.

:: --- STEP 1: DATA CLEANING (LAZADA) ---
ECHO [STEP 1/10] Cleaning Lazada Unlabeled Corpus...
python src/01_lazada_cleaning_dapt.py
IF %ERRORLEVEL% NEQ 0 GOTO ERROR

:: --- STEP 2: DATA CLEANING (SYNTHETIC) ---
ECHO.
ECHO [STEP 2/10] Cleaning Synthetic Corpus (Deduplication)...
python src/02_synthetic_cleaning_dapt.py
IF %ERRORLEVEL% NEQ 0 GOTO ERROR

:: --- STEP 3: HYBRID ASSEMBLY ---
ECHO.
ECHO [STEP 3/10] Assembling Hybrid Corpus (Lazada + Synthetic)...
python src/03_hybrid_assembly_dapt.py
IF %ERRORLEVEL% NEQ 0 GOTO ERROR

:: --- STEP 4: STRATIFICATION ---
ECHO.
ECHO [STEP 4/10] Stratifying FiReCS (80/10/10 Split)...
python src/04_firecs_stratification.py
IF %ERRORLEVEL% NEQ 0 GOTO ERROR

:: --- STEP 5: DAPT TRAINING ---
ECHO.
ECHO [STEP 5/10] Stage 1: Domain-Adaptive Pre-Training (Model B Base)...
ECHO (This will take time. Training on Hybrid Corpus...)
python src/05_train_stage1_dapt.py
IF %ERRORLEVEL% NEQ 0 GOTO ERROR

:: --- STEP 6: FINE-TUNING ---
ECHO.
ECHO [STEP 6/10] Stage 2: Supervised Fine-Tuning...
ECHO (Training Model A, Model B, and Model C sequentially...)
python src/06_train_stage2_finetune.py
IF %ERRORLEVEL% NEQ 0 GOTO ERROR

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
ECHO      EXPERIMENT COMPLETE! SUCCESS!
ECHO      1. Metrics: 'reports/metrics/final_metrics.csv'
ECHO      2. Figures: 'reports/figures/'
ECHO.
ECHO      To launch the Interactive Simulation App, run:
ECHO      streamlit run src/10_simulation_app.py
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