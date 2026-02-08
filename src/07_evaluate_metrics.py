import pandas as pd
import numpy as np
import torch
import time
import os
import sys
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import onnxruntime as ort

# Import Project Config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

def get_dir_size(path):
    """Calculates total size of a model directory in MB."""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if "checkpoint" not in fp:
                total_size += os.path.getsize(fp)
    return total_size / (1024 * 1024)

def evaluate_pytorch_model(model_path, test_df):
    """Evaluates standard Transformers models (Model A, B, C)."""
    print(f"   Loading {model_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.to("cpu") 
        model.eval()
    except Exception as e:
        print(f"   [ERROR] Could not load model: {e}")
        return None

    texts = test_df['review'].tolist()
    labels = test_df['label'].tolist()
    
    # 1. Predictive Performance (Batch Inference)
    print("   Running Prediction on Test Set...")
    # FIX: Force max_length to match ONNX workload for fairness
    inputs = tokenizer(texts, padding="max_length", truncation=True, 
                      max_length=config.MAX_LEN, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        preds = torch.argmax(outputs.logits, dim=-1).numpy()

    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
    acc = accuracy_score(labels, preds)

    # 2. Latency Benchmarking (Single Sequence)
    print("   Benchmarking Latency...")
    dummy_text = "Ang ganda ng product na ito sobrang sulit."
    dummy_input = tokenizer(dummy_text, padding="max_length", truncation=True, 
                           max_length=config.MAX_LEN, return_tensors="pt")
    
    latencies = []
    for _ in range(20): # Increased Warmup
        with torch.no_grad(): model(**dummy_input)
    
    for _ in range(100):
        start = time.perf_counter()
        with torch.no_grad(): model(**dummy_input)
        latencies.append((time.perf_counter() - start) * 1000)
    
    return {
        "Macro_F1": round(f1, 4), "Accuracy": round(acc, 4),
        "Latency_ms": round(np.mean(latencies), 2), "Model_Size_MB": round(get_dir_size(model_path), 2)
    }

def evaluate_onnx_model(model_dir, test_df):
    """Evaluates the Optimized ONNX model (Model D)."""
    onnx_path = os.path.join(model_dir, "model_quantized.onnx")
    print(f"   Loading ONNX: {onnx_path}...")
    
    if not os.path.exists(onnx_path): return None

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    # --- CRITICAL LATENCY FIX: Unlock Hardware Acceleration ---
    options = ort.SessionOptions()
    # 0 allows ORT to auto-manage threads for VNNI acceleration
    options.intra_op_num_threads = 0 
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(onnx_path, options, providers=["CPUExecutionProvider"])

    # 1. Predictive Performance
    texts = test_df['review'].tolist()
    labels = test_df['label'].tolist()
    preds = []

    for text in texts:
        inputs = tokenizer(text, return_tensors="np", padding="max_length", 
                          truncation=True, max_length=config.MAX_LEN)
        input_feed = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64)
        }
        outputs = session.run(None, input_feed)
        preds.append(np.argmax(outputs[0]))

    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
    acc = accuracy_score(labels, preds)

    # 2. Latency Benchmarking
    print("   Benchmarking ONNX Latency...")
    dummy_input = tokenizer("Ang ganda ng product na ito sobrang sulit.", 
                           return_tensors="np", padding="max_length", 
                           truncation=True, max_length=config.MAX_LEN)
    input_feed = {
        "input_ids": dummy_input["input_ids"].astype(np.int64),
        "attention_mask": dummy_input["attention_mask"].astype(np.int64)
    }
    
    latencies = []
    for _ in range(50): session.run(None, input_feed) # Stabilize Cache
    for _ in range(100):
        start = time.perf_counter()
        session.run(None, input_feed)
        latencies.append((time.perf_counter() - start) * 1000)

    return {
        "Macro_F1": round(f1, 4), "Accuracy": round(acc, 4),
        "Latency_ms": round(np.mean(latencies), 2),
        "Model_Size_MB": round(os.path.getsize(onnx_path) / (1024 * 1024), 2)
    }

def collect_metrics():
    print("\n--- STARTING FINAL METRICS COLLECTION ---")
    test_df = pd.read_csv(config.TEST_PATH)
    results = []

    models_to_test = [
        ("Model A", "Base DistilBERT", config.MODEL_A_DIR, "pytorch"),
        ("Model B", "DAPT-DistilBERT", config.MODEL_B_FINETUNED_DIR, "pytorch"),
        ("Model C", "XLM-RoBERTa", config.MODEL_C_DIR, "pytorch"),
        ("Model D", "Optimized DAPT", config.MODEL_D_DIR, "onnx")
    ]

    for m_id, m_name, m_path, m_type in models_to_test:
        print(f"\nEvaluating {m_name}...")
        if not os.path.exists(m_path): continue
        metrics = evaluate_pytorch_model(m_path, test_df) if m_type == "pytorch" else evaluate_onnx_model(m_path, test_df)
        if metrics:
            metrics.update({"Model_ID": m_id, "Model_Name": m_name})
            results.append(metrics)

    df_results = pd.DataFrame(results)
    # Speedup relative to the most powerful (but slowest) model, XLM-R
    base_lat = df_results.loc[df_results["Model_ID"] == "Model C", "Latency_ms"].values[0]
    df_results["Speedup_Factor"] = round(base_lat / df_results["Latency_ms"], 2)

    out_path = os.path.join(config.BASE_DIR, "results", "metrics_summary.csv")
    df_results.to_csv(out_path, index=False)
    print(f"\nSUCCESS! Results generated.\n{df_results.to_string(index=False)}")

if __name__ == "__main__":
    collect_metrics()