# FILE: 09_benchmark_simulation.py
import os
import sys
import time
import pandas as pd
import psutil
import gc
import torch
import platform
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from optimum.onnxruntime import ORTModelForSequenceClassification

# --- PATH CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)

DATA_PATH = os.path.join(BASE_DIR, 'data', '03_processed', 'FiReCS_Final', 'test.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports', 'metrics')
os.makedirs(REPORTS_DIR, exist_ok=True)

# Comparison List
MODELS_TO_BENCHMARK = [
    ("Model A", os.path.join(MODELS_DIR, "model_a_base"), "pytorch"),
    ("Model B", os.path.join(MODELS_DIR, "model_b_dapt"), "pytorch"),
    ("Model C", os.path.join(MODELS_DIR, "model_c_xlmr"), "pytorch"),
    ("Model D", os.path.join(MODELS_DIR, "model_d_onnx"), "onnx")
]

def get_dir_size_mb(path):
    total_size = 0
    if not os.path.exists(path):
        return 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if "checkpoint" not in fp:
                total_size += os.path.getsize(fp)
    return total_size / (1024 * 1024)

def load_tokenizer_safe(model_path):
    try:
        return AutoTokenizer.from_pretrained(model_path)
    except:
        return AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")

def run_stress_test(name, path, runtime, texts):
    print(f"\n>>> Stress Testing: {name} ({runtime})")
    
    # Garbage Collect
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 1. MEASURE COLD LOAD TIME (Academic Metric)
    start_load = time.time()
    tokenizer = load_tokenizer_safe(path)
    if runtime == "pytorch":
        model = AutoModelForSequenceClassification.from_pretrained(path)
        model.to("cpu")
        model.eval()
    else:
        model = ORTModelForSequenceClassification.from_pretrained(path, provider="CPUExecutionProvider")
    load_time = time.time() - start_load
    print(f"  Load Time: {load_time:.4f}s")

    # 2. WARMUP PHASE (Stabilize CPU Cache)
    # Run 10 dummy inferences to wake up the CPU cores
    warmup_text = texts[0]
    warmup_input = tokenizer(warmup_text, return_tensors="pt", truncation=True, max_length=128)
    if "distilbert" in name.lower(): warmup_input.pop("token_type_ids", None)
    
    # Ensure inputs on CPU
    if runtime == "pytorch":
        warmup_input = {k: v.to("cpu") for k, v in warmup_input.items()}

    for _ in range(10):
        _ = model(**warmup_input)
    print("  Warmup Complete.")

    # 3. LATENCY & STABILITY LOOP
    latencies = []
    
    # Limit to 1000 requests loop (wrap around texts if needed)
    LOOP_LIMIT = 1000
    
    print(f"  Running {LOOP_LIMIT} inference requests...")
    
    count = 0
    start_total = time.time()
    
    for i in range(LOOP_LIMIT):
        text = texts[i % len(texts)]
        
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        if "distilbert" in name.lower(): inputs.pop("token_type_ids", None)
        
        if runtime == "pytorch":
            inputs = {k: v.to("cpu") for k, v in inputs.items()}
            
        start_inf = time.perf_counter()
        if runtime == "pytorch":
            with torch.no_grad():
                _ = model(**inputs)
        else:
            _ = model(**inputs)
        end_inf = time.perf_counter()
        
        latencies.append((end_inf - start_inf) * 1000) # ms
        count += 1
        
    total_time = time.time() - start_total
    
    # Metrics Calculation
    avg_lat = np.mean(latencies)
    std_lat = np.std(latencies) # STABILITY METRIC
    p95_lat = np.percentile(latencies, 95)
    throughput = count / total_time
    
    storage_mb = get_dir_size_mb(path)
    mem_usage = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    
    print(f"  Avg Latency: {avg_lat:.2f}ms (±{std_lat:.2f}ms)")
    print(f"  Throughput: {throughput:.2f} req/sec")
    
    del model
    del tokenizer
    return {
        "Model Name": name,
        "Storage_MB": storage_mb,
        "Memory_MB": mem_usage,
        "Throughput_RPS": throughput,
        "Load_Time_s": load_time,
        "Latency_Mean_ms": avg_lat,
        "Latency_StdDev_ms": std_lat,
        "Latency_P95_ms": p95_lat
    }

def main():
    print(f"Project Root Detected: {BASE_DIR}")
    print("--- STAGE 9: ACADEMIC BENCHMARKING (CPU ONLY) ---")
    
    # Log Hardware Specs for Thesis
    print(f"[INFO] CPU: {platform.processor()}")
    print(f"[INFO] OS: {platform.system()} {platform.release()}")
    print(f"[INFO] Python: {sys.version.split()[0]}")

    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] Test data not found: {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)
    if 'review' in df.columns: df = df.rename(columns={'review': 'text'})
    texts = df['text'].tolist()[:100] # Use subset for speed
    
    results = []
    
    for name, path, runtime in MODELS_TO_BENCHMARK:
        if not os.path.exists(path):
            print(f"[SKIP] {name} path not found: {path}")
            continue
            
        stats = run_stress_test(name, path, runtime, texts)
        results.append(stats)
        
    if results:
        res_df = pd.DataFrame(results)
        out_path = os.path.join(REPORTS_DIR, "benchmark_results_academic.csv")
        res_df.to_csv(out_path, index=False)
        print(f"\nSUCCESS: Benchmarks saved to {out_path}")
        print(res_df.to_string(index=False))

if __name__ == "__main__":
    main()