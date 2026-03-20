# FILE: 08_evaluate_metrics.py
import os
import sys
import time
import gc
import pandas as pd
import numpy as np
import torch
import onnxruntime as ort
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, set_seed
from optimum.onnxruntime import ORTModelForSequenceClassification
import warnings

warnings.filterwarnings("ignore")

# --- ALIGN WITH TRAINING GUIDELINES (DETERMINISM) ---
set_seed(42)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
torch.use_deterministic_algorithms(True, warn_only=True)

# Force single-threaded CPU execution for absolute latency consistency
torch.set_num_threads(1)

# --- AGGRESSIVE COMPATIBILITY PATCHES ---
import transformers
import transformers.models.auto
class MockAutoModelForVision2Seq:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        raise NotImplementedError("Mock class for compatibility.")
setattr(transformers, "AutoModelForVision2Seq", MockAutoModelForVision2Seq)
setattr(transformers.models.auto, "AutoModelForVision2Seq", MockAutoModelForVision2Seq)

# --- PATH CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(SCRIPT_DIR)
BASE_DIR = os.path.dirname(SRC_DIR)

TEST_DATA_PATH = os.path.join(BASE_DIR, 'data', '03_processed', 'FiReCS_Final', 'test.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports', 'metrics')
os.makedirs(REPORTS_DIR, exist_ok=True)

MODELS_TO_TEST = [
    ("A", "Model A (Base DistilmBERT)", os.path.join(MODELS_DIR, "model_a_base"), "pytorch"),
    ("B", "Model B (DAPT-DistilmBERT)", os.path.join(MODELS_DIR, "model_b_dapt"), "pytorch"),
    ("C", "Model C (XLM-R Base)", os.path.join(MODELS_DIR, "model_c_xlmr"), "pytorch"),
    ("D", "Model D (Optimized DAPT)", os.path.join(MODELS_DIR, "model_d_onnx"), "onnx")
]

def normalize_columns(df):
    """Syncs data cleaning with the training pipeline to maximize score."""
    if 'text' not in df.columns and 'review' in df.columns:
        df = df.rename(columns={'review': 'text'})
    if 'label' not in df.columns and 'sentiment' in df.columns:
        df = df.rename(columns={'sentiment': 'label'})
    
    # CRITICAL: Remove NaNs that might have entered the test set
    df = df.dropna(subset=['text', 'label'])
    df['label'] = df['label'].astype(int)
    return df

def get_model_size_mb(path):
    total_size = 0
    if not os.path.exists(path): return 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp) and "checkpoint" not in fp:
                total_size += os.path.getsize(fp)
    return total_size / (1024 * 1024)

def load_tokenizer_safe(model_path):
    try:
        return AutoTokenizer.from_pretrained(model_path)
    except Exception:
        return AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")

def evaluate_pytorch_cpu(path, texts, name):
    """Forces PyTorch to use the CPU for fair hardware benchmarking."""
    tokenizer = load_tokenizer_safe(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    
    device = torch.device('cpu')
    model.to(device)
    model.eval()
    
    # Warmup
    warmup_inputs = tokenizer(texts[0], return_tensors="pt", truncation=True, padding='max_length', max_length=128)
    if "distilbert" in name.lower() and "token_type_ids" in warmup_inputs:
        warmup_inputs.pop("token_type_ids")
    warmup_inputs = {k: v.to(device) for k, v in warmup_inputs.items()}
    with torch.no_grad():
        _ = model(**warmup_inputs)

    preds = []
    latencies = []
    
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding='max_length', max_length=128)
        if "distilbert" in name.lower() and "token_type_ids" in inputs:
            inputs.pop("token_type_ids")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        start_time = time.perf_counter()
        with torch.no_grad():
            outputs = model(**inputs)
        end_time = time.perf_counter()
        
        logits = outputs.logits.numpy()
        preds.append(np.argmax(logits, axis=-1)[0])
        latencies.append((end_time - start_time) * 1000)
        
    return preds, latencies

def evaluate_onnx_cpu(path, texts, name):
    """Evaluates the quantized ONNX model using an optimized CPU runtime."""
    tokenizer = load_tokenizer_safe(path)
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 1
    sess_options.inter_op_num_threads = 1
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    model = ORTModelForSequenceClassification.from_pretrained(
        path, 
        provider="CPUExecutionProvider",
        session_options=sess_options
    )
    
    warmup_inputs = tokenizer(texts[0], return_tensors="pt", truncation=True, padding='max_length', max_length=128)
    if "distilbert" in name.lower() and "token_type_ids" in warmup_inputs:
        warmup_inputs.pop("token_type_ids")
    _ = model(**warmup_inputs)

    preds = []
    latencies = []
    
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding='max_length', max_length=128)
        if "distilbert" in name.lower() and "token_type_ids" in inputs:
            inputs.pop("token_type_ids")
            
        start_time = time.perf_counter()
        outputs = model(**inputs)
        end_time = time.perf_counter()
        
        logits = outputs.logits
        preds.append(np.argmax(logits, axis=-1)[0])
        latencies.append((end_time - start_time) * 1000)
        
    return preds, latencies

def main():
    print("--- STAGE 8: FINAL METRICS EVALUATION (FAIR CPU BENCHMARK) ---")
    
    if not os.path.exists(TEST_DATA_PATH):
        print(f"[ERROR] Test data not found: {TEST_DATA_PATH}")
        return

    # FIXED: Apply normalization before extracting lists to ensure scientific consistency
    df_raw = pd.read_csv(TEST_DATA_PATH)
    df = normalize_columns(df_raw)
    
    texts = df['text'].tolist()
    labels = df['label'].tolist()
    
    raw_results = []
    raw_latencies_log = {} 
    
    for model_id, name, path, runtime in MODELS_TO_TEST:
        print(f"\nEvaluating {name} strictly on CPU...")
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
       
        if not os.path.exists(path):
            print(f"  [SKIP] Model not found at {path}")
            continue
           
        try:
            if runtime == "pytorch":
                preds, latencies = evaluate_pytorch_cpu(path, texts, name)
            else:
                preds, latencies = evaluate_onnx_cpu(path, texts, name)
               
            acc = accuracy_score(labels, preds)
            f1 = f1_score(labels, preds, average='macro')
            
            avg_lat_overall = np.mean(latencies)
            lat_100 = np.mean(latencies[:100]) if len(latencies) >= 100 else avg_lat_overall
            lat_1000 = np.mean(latencies[:1000]) if len(latencies) >= 1000 else avg_lat_overall
            p95_lat = np.percentile(latencies, 95)
            size_mb = get_model_size_mb(path)
           
            print(f"  Acc: {acc:.4f} | Macro F1: {f1:.4f} | Avg CPU Lat: {avg_lat_overall:.2f}ms")
           
            raw_results.append({
                "Model ID": model_id,
                "Model Name": name,
                "Accuracy": round(acc, 4),
                "Macro F1 Score": round(f1, 4),
                "Avg Latency (100 runs) ms": round(lat_100, 2),
                "Avg Latency (1000 runs) ms": round(lat_1000, 2),
                "Avg Latency (Overall) ms": round(avg_lat_overall, 2),
                "P95 Latency (ms)": round(p95_lat, 2),
                "Model Size (MB)": round(size_mb, 2),
                "Runtime": "CPU_ONLY"
            })
            
            raw_latencies_log[name] = latencies
            
        except Exception as e:
            print(f"  [ERROR] Failed to evaluate {name}: {e}")

    if raw_results:
        res_df = pd.DataFrame(raw_results)
        
        if "Avg Latency (Overall) ms" in res_df.columns and len(res_df) > 0:
            base_lat = res_df.iloc[0]["Avg Latency (Overall) ms"]
            res_df["Speedup Factor"] = round(base_lat / res_df["Avg Latency (Overall) ms"], 2)

        main_cols = [
            "Model ID", "Model Name", "Accuracy", "Macro F1 Score", 
            "Avg Latency (Overall) ms", "P95 Latency (ms)", 
            "Model Size (MB)", "Runtime", "Speedup Factor"
        ]
        main_path = os.path.join(REPORTS_DIR, "final_metrics.csv")
        res_df[main_cols].to_csv(main_path, index=False)
        
        stability_cols = [
            "Model Name", "Avg Latency (100 runs) ms", 
            "Avg Latency (1000 runs) ms", "Avg Latency (Overall) ms"
        ]
        stability_path = os.path.join(REPORTS_DIR, "latency_stability.csv")
        res_df[stability_cols].to_csv(stability_path, index=False)
        
        if raw_latencies_log:
            max_len = max(len(l) for l in raw_latencies_log.values())
            padded_log = {k: v + [np.nan] * (max_len - len(v)) for k, v in raw_latencies_log.items()}
            raw_lat_df = pd.DataFrame(padded_log)
            raw_lat_path = os.path.join(REPORTS_DIR, "raw_inference_latencies.csv")
            raw_lat_df.to_csv(raw_lat_path, index=False)
            print(f"SUCCESS: Raw latency distributions saved to {raw_lat_path}")

        print(f"SUCCESS: Main evaluation saved to {main_path}")
        print(f"SUCCESS: Stability evaluation saved to {stability_path}")

if __name__ == "__main__":
    main()