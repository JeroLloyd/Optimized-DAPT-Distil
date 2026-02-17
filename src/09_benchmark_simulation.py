import os
import sys
import time
import pandas as pd
import psutil
import gc
import torch
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
            if "checkpoint-" not in fp: 
                total_size += os.path.getsize(fp)
    return total_size / (1024 * 1024)

def load_tokenizer_safe(path):
    try:
        return AutoTokenizer.from_pretrained(path)
    except Exception as e:
        print(f"  [WARN] Tokenizer failed for {path}. Using fallback...")
        if "xlm" in str(path).lower():
            return AutoTokenizer.from_pretrained("xlm-roberta-base")
        return AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")

def run_stress_test(name, path, runtime, texts):
    print(f"Benchmarking {name}...")
    
    # 1. Clean Slate
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    time.sleep(1)
    
    # 2. Storage
    storage_mb = get_dir_size_mb(path)
    
    # 3. Memory & Load
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / 1024 / 1024
    
    start_load = time.time()
    try:
        tokenizer = load_tokenizer_safe(path)
        
        if runtime == "pytorch":
            model = AutoModelForSequenceClassification.from_pretrained(path)
            model.to("cpu") # FORCE CPU
            model.eval()
        else:
            model = ORTModelForSequenceClassification.from_pretrained(
                path, 
                provider="CPUExecutionProvider" # FORCE CPU
            )
    except Exception as e:
        print(f"  [ERROR] Failed to load {name}: {e}")
        return None

    load_time = time.time() - start_load
    mem_after = process.memory_info().rss / 1024 / 1024
    mem_usage = mem_after - mem_before
    
    # 4. Throughput
    bench_texts = (texts * (1000 // len(texts) + 1))[:1000]
    
    print(f"  Storage: {storage_mb:.2f} MB | Memory: {mem_usage:.2f} MB | Load: {load_time:.2f}s")
    
    start_inf = time.time()
    count = 0
    
    if hasattr(model, "config"):
        model.config.use_cache = False

    is_distilbert = "distilbert" in str(path).lower() or "model_a" in str(path).lower() or "model_b" in str(path).lower()

    for text in bench_texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        
        if is_distilbert:
            inputs.pop("token_type_ids", None)
        
        # Ensure inputs are CPU
        inputs = {k: v.to("cpu") for k, v in inputs.items()}
            
        _ = model(**inputs)
        count += 1
        
    total_time = time.time() - start_inf
    throughput = count / total_time
    print(f"  Throughput: {throughput:.2f} req/sec")
    
    del model
    del tokenizer
    
    return {
        "Model Name": name,
        "Storage_MB": storage_mb,
        "Memory_MB": mem_usage,
        "Throughput_RPS": throughput,
        "Load_Time_s": load_time
    }

def main():
    print(f"Project Root Detected: {BASE_DIR}")
    print("--- STAGE 5: SYSTEM BENCHMARKING (CPU ONLY) ---")

    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] Test data not found: {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)
    if 'review' in df.columns: df = df.rename(columns={'review': 'text'})
    texts = df['text'].tolist()[:100]
    
    results = []
    
    for name, path, runtime in MODELS_TO_BENCHMARK:
        if not os.path.exists(path):
            print(f"[SKIP] {name} path not found: {path}")
            continue
            
        stats = run_stress_test(name, path, runtime, texts)
        if stats:
            results.append(stats)

    if results:
        out_path = os.path.join(REPORTS_DIR, 'benchmark_results.csv')
        pd.DataFrame(results).to_csv(out_path, index=False)
        print(f"SUCCESS: Real benchmark data saved to {out_path}")
    else:
        print("[WARNING] No benchmark results generated.")

if __name__ == "__main__":
    main()