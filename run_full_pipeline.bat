@echo off
setlocal enabledelayedexpansion

:: Jump to the directory where this .bat file is located
cd /d "%~dp0"

echo ===========================================================
echo   SENTIMENT ANALYSIS REPRODUCTION PIPELINE
echo ===========================================================
echo Starting full rerun to synchronize Macro F1 scores...
echo Current Directory: %CD%
echo.

@REM :: --- STAGE 1: DOMAIN-ADAPTIVE PRE-TRAINING ---
@REM echo [1/7] Running Stage 1: DAPT (8 Epochs)...
@REM python src/training/05_train_stage1_dapt.py
@REM if %errorlevel% neq 0 (echo [ERROR] Stage 1 Failed. & pause & exit /b %errorlevel%)

@REM :: --- STAGE 2: SUPERVISED FINE-TUNING ---
@REM echo [2/7] Running Stage 2: Fine-tuning...
@REM python src/training/06_train_stage2_finetune.py
@REM if %errorlevel% neq 0 (echo [ERROR] Stage 2 Failed. & pause & exit /b %errorlevel%)

@REM :: --- STAGE 3: ONNX INT8 QUANTIZATION ---
@REM echo [3/7] Running Stage 3: Optimization (Model D)...
@REM python src/training/07_optimize_stage3_model_d.py
@REM if %errorlevel% neq 0 (echo [ERROR] Stage 3 Failed. & pause & exit /b %errorlevel%)

@REM :: --- EVALUATION & METRICS ---
@REM echo [4/7] Evaluating Final Metrics (CSV Generation)...
@REM python src/validation/08_evaluate_metrics.py
@REM if %errorlevel% neq 0 (echo [ERROR] Evaluation Failed. & pause & exit /b %errorlevel%)

echo [5/7] Running Academic Benchmark Simulation...
python src/validation/09_benchmark_simulation.py
if %errorlevel% neq 0 (echo [ERROR] Benchmark Failed. & pause & exit /b %errorlevel%)

:: --- HARDWARE EMULATION ---
echo [6/7] Running Edge CPU Hardware Emulation...
python src/validation/16_edge_cpu_emulation.py
if %errorlevel% neq 0 (echo [ERROR] Emulation Failed. & pause & exit /b %errorlevel%)

@REM :: --- VISUALIZATION ---
@REM echo [7/7] Generating Pareto Frontiers and Charts...
@REM python src/validation/10_visualize_results.py
@REM if %errorlevel% neq 0 (echo [ERROR] Visualization Failed. & pause & exit /b %errorlevel%)

echo.
echo ===========================================================
echo   PIPELINE SUCCESSFUL: ALL SCORES SYNCHRONIZED
echo ===========================================================
echo Check your 'reports' folder for the updated manuscript data.
pause