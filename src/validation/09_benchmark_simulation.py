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
import onnxruntime as ort
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from optimum.onnxruntime import ORTModelForSequenceClassification

# --- PATH CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(SCRIPT_DIR)
BASE_DIR = os.path.dirname(SRC_DIR)

DATA_PATH = os.path.join(BASE_DIR, 'data', '03_processed', 'FiReCS_Final', 'test.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports', 'metrics')
os.makedirs(REPORTS_DIR, exist_ok=True)

# Comparison List
MODELS_TO_BENCHMARK = [
    ("Model A (Base DistilmBERT)", os.path.join(MODELS_DIR, "model_a_base"), "pytorch"),
    ("Model B (DAPT-DistilmBERT)", os.path.join(MODELS_DIR, "model_b_dapt"), "pytorch"),
    ("Model C (XLM-R Base)", os.path.join(MODELS_DIR, "model_c_xlmr"), "pytorch"),
    ("Model D (Optimized DAPT)", os.path.join(MODELS_DIR, "model_d_onnx"), "onnx")
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

def run_stress_test(name, path, runtime, texts, device="cpu"):
    print(f"\n>>> Stress Testing: {name} ({runtime} on {device.upper()})")
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- ALIGNMENT: Enforce single-threading on CPU for fair baseline comparison ---
    if device == "cpu":
        torch.set_num_threads(1)

    start_load = time.time()
    tokenizer = load_tokenizer_safe(path)
    
    if runtime == "pytorch":
        model = AutoModelForSequenceClassification.from_pretrained(path)
        model.to(device)
        model.eval()
    else:
        available_providers = ort.get_available_providers()
        print(f"  [INFO] Available ONNX Providers: {available_providers}")
        
        provider = "CUDAExecutionProvider" if device == "cuda" else "CPUExecutionProvider"
        
        if provider == "CUDAExecutionProvider" and provider not in available_providers:
            print(f"  [WARN] {provider} not found in ONNX Runtime. Falling back to CPUExecutionProvider.")
            provider = "CPUExecutionProvider"
            
        # --- ALIGNMENT: Apply strict ONNX graph optimizations ---
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if device == "cpu":
            sess_options.intra_op_num_threads = 1
            sess_options.inter_op_num_threads = 1
            
        model = ORTModelForSequenceClassification.from_pretrained(
            path, 
            provider=provider,
            session_options=sess_options
        )
        
    load_time = time.time() - start_load
    print(f"  Load Time: {load_time:.4f}s")

    warmup_text = texts[0]
    warmup_input = tokenizer(warmup_text, return_tensors="pt", truncation=True, padding='max_length', max_length=128)
    if "distilmbert" in name.lower(): 
        warmup_input.pop("token_type_ids", None)
    
    if runtime == "pytorch":
        warmup_input = {k: v.to(device) for k, v in warmup_input.items()}

    for _ in range(10):
        _ = model(**warmup_input)
    print("  Warmup Complete.")

    latencies = []
    LOOP_LIMIT = 1000
    print(f"  Running {LOOP_LIMIT} inference requests...")
    
    count = 0
    start_total = time.time()
    
    for i in range(LOOP_LIMIT):
        text = texts[i % len(texts)]
        
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding='max_length', max_length=128)
        if "distilbert" in name.lower(): 
            inputs.pop("token_type_ids", None)
        
        if runtime == "pytorch":
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
        start_inf = time.perf_counter()
        if runtime == "pytorch":
            with torch.no_grad():
                _ = model(**inputs)
        else:
            _ = model(**inputs)
        end_inf = time.perf_counter()
        
        latencies.append((end_inf - start_inf) * 1000)
        count += 1
        
    total_time = time.time() - start_total
    
    avg_lat = np.mean(latencies)
    std_lat = np.std(latencies)
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
        "Device": device.upper(),
        "Empirical Space Complexity (Storage MB)": round(storage_mb, 2),
        "Empirical Space Complexity (RAM MB)": round(mem_usage, 2),
        "Throughput (req/sec)": round(throughput, 2),
        "Load Time (s)": round(load_time, 4),
        "Empirical Time Complexity (Mean ms)": round(avg_lat, 2),
        "Empirical Time Complexity (StdDev ms)": round(std_lat, 2),
        "Empirical Time Complexity (P95 ms)": round(p95_lat, 2)
    }

def main():
    print(f"Project Root Detected: {BASE_DIR}")
    print("--- STAGE 9: ACADEMIC BENCHMARKING (CPU & GPU) ---")
    
    print(f"[INFO] CPU: {platform.processor()}")
    print(f"[INFO] OS: {platform.system()} {platform.release()}")
    print(f"[INFO] Python: {sys.version.split()[0]}")
    
    devices_to_test = ["cpu"]
    if torch.cuda.is_available():
        devices_to_test.append("cuda")
        print(f"[INFO] PyTorch GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("[WARN] PyTorch CUDA not available. Benchmarking on CPU only.")

    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] Test data not found: {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)
    if 'review' in df.columns: 
        df = df.rename(columns={'review': 'text'})
    texts = df['text'].tolist()
    
    results = []
    
    for device in devices_to_test:
        for name, path, runtime in MODELS_TO_BENCHMARK:
            if not os.path.exists(path):
                print(f"[SKIP] {name} path not found: {path}")
                continue
                
            stats = run_stress_test(name, path, runtime, texts, device=device)
            results.append(stats)
        
    if results:
        res_df = pd.DataFrame(results)
        out_path = os.path.join(REPORTS_DIR, "benchmark_results_academic.csv")
        res_df.to_csv(out_path, index=False)
        print(f"\nSUCCESS: Benchmarks saved to {out_path}")
        print(res_df.to_string(index=False))

if __name__ == "__main__":
    main()