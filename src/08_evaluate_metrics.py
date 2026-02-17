import os
import sys
import pandas as pd
import numpy as np
import time
import torch
import gc
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from optimum.onnxruntime import ORTModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score

# --- PATH CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)

DATA_PATH = os.path.join(BASE_DIR, 'data', '03_processed', 'FiReCS_Final', 'test.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'reports', 'metrics')
os.makedirs(RESULTS_DIR, exist_ok=True)

# The 4 Models from the Manuscript
MODELS_TO_TEST = [
    ("Model A", "Base DistilBERT", os.path.join(MODELS_DIR, "model_a_base"), "pytorch"),
    ("Model B", "DAPT-DistilBERT", os.path.join(MODELS_DIR, "model_b_dapt"), "pytorch"),
    ("Model C", "XLM-R Base", os.path.join(MODELS_DIR, "model_c_xlmr"), "pytorch"),
    ("Model D", "Optimized DAPT", os.path.join(MODELS_DIR, "model_d_onnx"), "onnx")
]

def get_model_size_mb(model_path):
    """Calculates total size of the model directory in MB."""
    total_size = 0
    if not os.path.exists(model_path):
        return 0
    for dirpath, _, filenames in os.walk(model_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if "checkpoint" in fp: 
                continue
            total_size += os.path.getsize(fp)
    return total_size / (1024 * 1024)

def load_tokenizer_safe(model_path, model_name):
    """Robust tokenizer loader with fallback."""
    try:
        return AutoTokenizer.from_pretrained(model_path)
    except Exception as e:
        print(f"  [WARN] Local tokenizer failed for {model_name}. Attempting fallback...")
        if "xlm-r" in model_name.lower():
            return AutoTokenizer.from_pretrained("xlm-roberta-base")
        elif "distilbert" in model_name.lower():
            return AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")
        raise e

def evaluate_pytorch(model_path, texts, model_name):
    tokenizer = load_tokenizer_safe(model_path, model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    # FORCE CPU
    model.to("cpu")
    model.eval()
    
    preds = []
    latencies = []
    
    # Warmup
    warmup_inputs = tokenizer(texts[0], return_tensors="pt", truncation=True, max_length=128)
    if "distilbert" in model_name.lower():
        warmup_inputs.pop("token_type_ids", None)
    _ = model(**warmup_inputs)

    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        if "distilbert" in model_name.lower():
            inputs.pop("token_type_ids", None)
        
        # Ensure inputs are on CPU
        inputs = {k: v.to("cpu") for k, v in inputs.items()}
        
        start = time.perf_counter()
        with torch.no_grad():
            outputs = model(**inputs)
        end = time.perf_counter()
        
        latencies.append((end - start) * 1000) # ms
        preds.append(np.argmax(outputs.logits.detach().numpy(), axis=1)[0])
    
    del model
    del tokenizer
    return preds, latencies

def evaluate_onnx(model_path, texts, model_name):
    tokenizer = load_tokenizer_safe(model_path, model_name)
    
    # FORCE CPU PROVIDER
    model = ORTModelForSequenceClassification.from_pretrained(
        model_path, 
        provider="CPUExecutionProvider"
    )
    
    preds = []
    latencies = []
    
    # Warmup
    warmup_inputs = tokenizer(texts[0], return_tensors="pt", truncation=True, max_length=128)
    if "distilbert" in model_name.lower():
        warmup_inputs.pop("token_type_ids", None)
    _ = model(**warmup_inputs)

    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        if "distilbert" in model_name.lower():
            inputs.pop("token_type_ids", None)
        
        # Ensure inputs are on CPU
        inputs = {k: v.to("cpu") for k, v in inputs.items()}
        
        start = time.perf_counter()
        outputs = model(**inputs)
        end = time.perf_counter()
        
        latencies.append((end - start) * 1000) # ms
        preds.append(np.argmax(outputs.logits, axis=1)[0])
        
    del model
    del tokenizer
    return preds, latencies

def main():
    print(f"Project Root: {BASE_DIR}")
    print("--- STAGE 4: EVALUATION (CPU ONLY) ---")
    
    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] Test data not found: {DATA_PATH}")
        return
        
    df = pd.read_csv(DATA_PATH)
    if 'review' in df.columns: df = df.rename(columns={'review': 'text'})
    
    texts = df['text'].tolist()
    labels = df['label'].tolist()
    
    raw_results = []
    
    for model_id, name, path, runtime in MODELS_TO_TEST:
        print(f"Evaluating {name} ({runtime})...")
        
        # Cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if not os.path.exists(path):
            print(f"  [SKIP] Model not found at {path}")
            continue
            
        try:
            if runtime == "pytorch":
                preds, latencies = evaluate_pytorch(path, texts, name)
            else:
                preds, latencies = evaluate_onnx(path, texts, name)
                
            acc = accuracy_score(labels, preds)
            f1 = f1_score(labels, preds, average='macro') 
            avg_lat = np.mean(latencies)
            p95_lat = np.percentile(latencies, 95)
            size_mb = get_model_size_mb(path)
            
            print(f"  Acc: {acc:.4f} | Macro F1: {f1:.4f} | Lat: {avg_lat:.2f}ms | Size: {size_mb:.2f}MB")
            
            raw_results.append({
                "Model ID": model_id,
                "Model Name": name,
                "Accuracy": acc,
                "Macro F1 Score": f1, 
                "Avg Latency (ms)": avg_lat,
                "P95 Latency (ms)": p95_lat,
                "Model Size (MB)": size_mb,
                "Runtime": runtime
            })
            
        except Exception as e:
            print(f"  [ERROR] Failed {model_id}: {e}")
            
    if raw_results:
        df_res = pd.DataFrame(raw_results)
        
        # Speedup Baseline
        baseline_row = df_res[df_res["Model ID"] == "Model A"]
        if not baseline_row.empty:
            baseline_lat = baseline_row.iloc[0]["Avg Latency (ms)"]
            df_res["Speedup Factor"] = baseline_lat / df_res["Avg Latency (ms)"]
        else:
            df_res["Speedup Factor"] = 1.0

        output_file = os.path.join(RESULTS_DIR, "final_metrics.csv")
        df_res.to_csv(output_file, index=False)
        print("\nFinal Metrics Table:")
        print(df_res[["Model Name", "Macro F1 Score", "Avg Latency (ms)", "Model Size (MB)", "Speedup Factor"]].to_string(index=False))
    else:
        print("[WARNING] No results generated.")

if __name__ == "__main__":
    main()