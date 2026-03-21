@echo off
setlocal enabledelayedexpansion

:: Jump to the directory where this .bat file is located
cd /d "%~dp0"

:: Initialize the timer using Python
echo import time > start_timer.py
echo with open("timer_temp.txt", "w") as f: >> start_timer.py
echo     f.write(str(time.time())) >> start_timer.py
python start_timer.py

echo ===========================================================
echo   SENTIMENT ANALYSIS REPRODUCTION PIPELINE
echo ===========================================================
echo Starting full rerun to synchronize Macro F1 scores...
echo Current Directory: %CD%
echo.

@REM :: --- STAGE 1: DOMAIN-ADAPTIVE PRE-TRAINING ---
@REM echo [1/12] Running Stage 1: DAPT...
@REM python src/training/05_train_stage1_dapt.py
@REM if %errorlevel% neq 0 (echo [ERROR] Stage 1 Failed. & pause & exit /b %errorlevel%)

@REM :: --- STAGE 2: SUPERVISED FINE-TUNING ---
@REM echo [2/12] Running Stage 2: Fine-tuning...
@REM python src/training/06_train_stage2_finetune.py
@REM if %errorlevel% neq 0 (echo [ERROR] Stage 2 Failed. & pause & exit /b %errorlevel%)

@REM :: --- STAGE 3: ONNX INT8 QUANTIZATION ---
@REM echo [3/12] Running Stage 3: Optimization (Model D)...
@REM python src/training/07_optimize_stage3_model_d.py
@REM if %errorlevel% neq 0 (echo [ERROR] Stage 3 Failed. & pause & exit /b %errorlevel%)

@REM :: --- ABLATION STUDY ---
@REM echo [4/12] Running Stage 4: Ablation Study Training...
@REM python src/validation/13_ablation_study.py
@REM if %errorlevel% neq 0 (echo [ERROR] Ablation Study Failed. & pause & exit /b %errorlevel%)

@REM :: --- EVALUATION & METRICS ---
@REM echo [5/12] Evaluating Final Metrics (CSV Generation)...
@REM python src/validation/08_evaluate_metrics.py
@REM if %errorlevel% neq 0 (echo [ERROR] Evaluation Failed. & pause & exit /b %errorlevel%)

@REM echo [6/12] Running Academic Benchmark Simulation...
@REM python src/validation/09_benchmark_simulation.py
@REM if %errorlevel% neq 0 (echo [ERROR] Benchmark Failed. & pause & exit /b %errorlevel%)

@REM :: --- HARDWARE EMULATION ---
@REM echo [7/12] Running Edge CPU Hardware Emulation...
@REM python src/validation/16_edge_cpu_emulation.py
@REM if %errorlevel% neq 0 (echo [ERROR] Emulation Failed. & pause & exit /b %errorlevel%)

:: --- STATISTICAL VALIDATION ---
echo [8/12] Running Multi-Seed Variance Test...
python src/validation/14_point4_validation.py
if %errorlevel% neq 0 (echo [ERROR] Variance Test Failed. & pause & exit /b %errorlevel%)

@REM echo [9/12] Running Bootstrap Hypothesis Test...
@REM python src/validation/15_point7_bootstrap.py
@REM if %errorlevel% neq 0 (echo [ERROR] Bootstrap Test Failed. & pause & exit /b %errorlevel%)

@REM :: --- ERROR ANALYSIS ---
@REM echo [10/12] Generating Error Analysis (Models A, B, Ablation)...
@REM python src/validation/12_error_analysis.py
@REM if %errorlevel% neq 0 (echo [ERROR] Error Analysis 1 Failed. & pause & exit /b %errorlevel%)

@REM echo [11/12] Generating Error Analysis (Models C, D)...
@REM python src/validation/12_error_analysis_model_C_D.py
@REM if %errorlevel% neq 0 (echo [ERROR] Error Analysis 2 Failed. & pause & exit /b %errorlevel%)

@REM :: --- VISUALIZATION ---
@REM echo [12/12] Generating Pareto Frontiers and Charts...
@REM python src/validation/10_visualize_results.py
@REM if %errorlevel% neq 0 (echo [ERROR] Visualization Failed. & pause & exit /b %errorlevel%)

echo.
echo ===========================================================
echo   PIPELINE SUCCESSFUL: ALL SCORES SYNCHRONIZED
echo ===========================================================

:: Calculate and display the total execution time
echo import time > end_timer.py
echo import os >> end_timer.py
echo with open("timer_temp.txt", "r") as f: >> end_timer.py
echo     start_time = float(f.read()) >> end_timer.py
echo diff = time.time() - start_time >> end_timer.py
echo m, s = divmod(diff, 60) >> end_timer.py
echo h, m = divmod(m, 60) >> end_timer.py
echo print(f"Total Execution Time: {int(h)} hours {int(m)} minutes {int(s)} seconds") >> end_timer.py
echo os.remove("timer_temp.txt") >> end_timer.py
echo os.remove("start_timer.py") >> end_timer.py
echo os.remove("end_timer.py") >> end_timer.py
python end_timer.py

echo.
echo Check your 'reports' folder for the updated manuscript data.
pause