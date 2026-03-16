# FILE: 08_evaluate_metrics.py
import os
import sys
import time
import gc
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, set_seed
from optimum.onnxruntime import ORTModelForSequenceClassification
import warnings

warnings.filterwarnings("ignore")

# --- ALIGN WITH TRAINING GUIDELINES (DETERMINISM) ---
set_seed(42)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
torch.use_deterministic_algorithms(True, warn_only=True)

# --- SUPER-AGGRESSIVE MONKEY PATCH FOR TRANSFORMERS ---
import transformers
import transformers.models.auto

class MockAutoModelForVision2Seq:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        raise NotImplementedError("This is a mock class for compatibility.")

setattr(transformers, "AutoModelForVision2Seq", MockAutoModelForVision2Seq)
setattr(transformers.models.auto, "AutoModelForVision2Seq", MockAutoModelForVision2Seq)

if "transformers" in sys.modules:
    sys.modules["transformers"].AutoModelForVision2Seq = MockAutoModelForVision2Seq

try:
    if hasattr(transformers, "_import_structure"):
        transformers._import_structure["models.auto"].append("AutoModelForVision2Seq")
    if hasattr(transformers, "_class_to_module"):
        transformers._class_to_module["AutoModelForVision2Seq"] = "models.auto"
except Exception:
    pass
# ------------------------------------------------------

# --- PATH CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(SCRIPT_DIR)
BASE_DIR = os.path.dirname(SRC_DIR)

TEST_DATA_PATH = os.path.join(BASE_DIR, 'data', '03_processed', 'FiReCS_Final', 'test.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports', 'metrics')
os.makedirs(REPORTS_DIR, exist_ok=True)

MODELS_TO_TEST = [
    ("A", "Model A (Base DistilBERT)", os.path.join(MODELS_DIR, "model_a_base"), "pytorch"),
    ("B", "Model B (DAPT-DistilBERT)", os.path.join(MODELS_DIR, "model_b_dapt"), "pytorch"),
    ("C", "Model C (XLM-R Base)", os.path.join(MODELS_DIR, "model_c_xlmr"), "pytorch"),
    ("D", "Model D (Optimized DAPT)", os.path.join(MODELS_DIR, "model_d_onnx"), "onnx")
]

def get_model_size_mb(path):
    total_size = 0
    if not os.path.exists(path):
        return 0
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

def evaluate_pytorch(path, texts, name):
    tokenizer = load_tokenizer_safe(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
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
        
        logits = outputs.logits.cpu().numpy()
        preds.append(np.argmax(logits, axis=-1)[0])
        latencies.append((end_time - start_time) * 1000)
        
    return preds, latencies

def evaluate_onnx(path, texts, name):
    tokenizer = load_tokenizer_safe(path)
    model = ORTModelForSequenceClassification.from_pretrained(path, provider="CPUExecutionProvider")
    
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
    print("--- STAGE 8: FINAL METRICS EVALUATION (DETERMINISTIC) ---")
    
    if not os.path.exists(TEST_DATA_PATH):
        print(f"[ERROR] Test data not found: {TEST_DATA_PATH}")
        return

    df = pd.read_csv(TEST_DATA_PATH)
    texts = df['review'].tolist() if 'review' in df.columns else df['text'].tolist()
    labels = df['label'].tolist()
    
    raw_results = []
    
    for model_id, name, path, runtime in MODELS_TO_TEST:
        print(f"\nEvaluating {name}...")
        
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
                "Accuracy": round(acc, 4),
                "Macro F1 Score": round(f1, 4),
                "Avg Latency (ms)": round(avg_lat, 2),
                "P95 Latency (ms)": round(p95_lat, 2),
                "Model Size (MB)": round(size_mb, 2),
                "Runtime": runtime.upper()
            })
            
        except Exception as e:
            print(f"  [ERROR] Failed to evaluate {name}: {e}")

    if raw_results:
        res_df = pd.DataFrame(raw_results)
        
        if "Avg Latency (ms)" in res_df.columns and len(res_df) > 0:
            base_lat = res_df.iloc[0]["Avg Latency (ms)"]
            res_df["Speedup Factor"] = round(base_lat / res_df["Avg Latency (ms)"], 2)

        out_path = os.path.join(REPORTS_DIR, "final_metrics.csv")
        res_df.to_csv(out_path, index=False)
        print(f"\nSUCCESS: Results saved to {out_path}")

if __name__ == "__main__":
    main()