import os
import sys
import time
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import onnxruntime as ort
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from optimum.onnxruntime import ORTModelForSequenceClassification
from sklearn.metrics import f1_score
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# --- AGGRESSIVE COMPATIBILITY PATCHES ---
import transformers
import transformers.models.auto
class MockAutoModelForVision2Seq:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        raise NotImplementedError("Mock class for compatibility.")
setattr(transformers, "AutoModelForVision2Seq", MockAutoModelForVision2Seq)
setattr(transformers.models.auto, "AutoModelForVision2Seq", MockAutoModelForVision2Seq)
if not hasattr(transformers.utils, 'is_offline_mode'):
    transformers.utils.is_offline_mode = lambda: False

# --- PATH CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = SCRIPT_DIR if os.path.basename(SCRIPT_DIR) != "src" else os.path.dirname(SCRIPT_DIR)

DATA_PATH = os.path.join(BASE_DIR, 'data', '03_processed', 'FiReCS_Final', 'test.csv')
MODEL_A_DIR = os.path.join(BASE_DIR, 'models', 'model_a_base')
MODEL_B_DIR = os.path.join(BASE_DIR, 'models', 'model_b_dapt')
MODEL_C_DIR = os.path.join(BASE_DIR, 'models', 'model_c_xlmr')
MODEL_D_DIR = os.path.join(BASE_DIR, 'models', 'model_d_onnx')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports', 'metrics')
FIGURES_DIR = os.path.join(BASE_DIR, 'reports', 'figures')

os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

def safe_load_tokenizer(model_path, fallback_name):
    try:
        return AutoTokenizer.from_pretrained(model_path)
    except Exception:
        return AutoTokenizer.from_pretrained(fallback_name)

def benchmark_pytorch_edge(model_path, texts, true_labels, cores, fallback_tokenizer):
    torch.set_num_threads(cores)
    
    tokenizer = safe_load_tokenizer(model_path, fallback_tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to("cpu")
    model.eval()
    
    warmup = tokenizer(texts[0], return_tensors="pt", truncation=True, max_length=128)
    if "distilbert" in fallback_tokenizer:
        warmup.pop("token_type_ids", None)
    
    with torch.no_grad():
        _ = model(**warmup)

    latencies = []
    predictions = []
    
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        if "distilbert" in fallback_tokenizer:
            inputs.pop("token_type_ids", None)
        
        start = time.perf_counter()
        with torch.no_grad():
            outputs = model(**inputs)
        latencies.append((time.perf_counter() - start) * 1000)
        
        logits = outputs.logits[0].cpu().numpy()
        predictions.append(np.argmax(logits))
        
    macro_f1 = f1_score(true_labels, predictions, average='macro')
    return np.mean(latencies), macro_f1

def benchmark_onnx_edge(model_path, texts, true_labels, cores):
    tokenizer = safe_load_tokenizer(model_path, "distilbert-base-multilingual-cased")
    
    options = ort.SessionOptions()
    options.intra_op_num_threads = cores
    options.inter_op_num_threads = 1 
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    model = ORTModelForSequenceClassification.from_pretrained(
        model_path, 
        provider="CPUExecutionProvider",
        session_options=options
    )
    
    warmup = tokenizer(texts[0], return_tensors="pt", truncation=True, max_length=128)
    warmup.pop("token_type_ids", None)
    _ = model(**warmup)

    latencies = []
    predictions = []
    
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        inputs.pop("token_type_ids", None)
        
        start = time.perf_counter()
        outputs = model(**inputs)
        latencies.append((time.perf_counter() - start) * 1000)
        
        logits = outputs.logits[0]
        predictions.append(np.argmax(logits))
        
    macro_f1 = f1_score(true_labels, predictions, average='macro')
    return np.mean(latencies), macro_f1

def generate_visualizations(df):
    sns.set_theme(style="whitegrid", rc={"axes.facecolor": "#f8f9fa"})
    palette = ["#7f8c8d", "#3498db", "#e74c3c", "#2ecc71"]
    
    # Image 1: Latency Comparison
    plt.figure(figsize=(14, 7))
    sns.barplot(data=df, x='Emulated_Device', y='Latency_ms', hue='Model', palette=palette)
    plt.title("Inference Latency Across Edge Devices", fontsize=16, fontweight='bold', pad=15)
    plt.ylabel("Average Latency (ms)", fontsize=12, fontweight='bold')
    plt.xlabel("Emulated Hardware Profile", fontsize=12, fontweight='bold')
    plt.legend(title=None, bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "8a_edge_latency.png"), dpi=300)
    plt.close()

    # Image 2: Efficiency Trade-off (Faceted Small Multiples)
    g = sns.relplot(
        data=df,
        x='Latency_ms',
        y='Macro_F1',
        hue='Model',
        col='Emulated_Device',
        col_wrap=2,
        kind='scatter',
        s=300,
        palette=palette,
        height=4.5,
        aspect=1.3,
        edgecolor='black',
        alpha=0.8
    )
    
    g.fig.suptitle("Edge Efficiency: Latency vs. Accuracy", fontsize=16, fontweight='bold', y=1.05)
    g.set_axis_labels("Inference Latency (ms) [Lower is Better]", "Macro F1 Score [Higher is Better]", fontweight='bold')
    g.set_titles(col_template="{col_name}", size=13, weight='bold')
    sns.move_legend(g, "upper center", bbox_to_anchor=(0.5, -0.05), ncol=4, title=None, frameon=True)
    
    plt.savefig(os.path.join(FIGURES_DIR, "8b_edge_efficiency.png"), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print("=== EDGE HARDWARE EMULATION BENCHMARK ===")
    
    torch.set_num_interop_threads(1)
    
    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] Test data not found: {DATA_PATH}")
        return
        
    df = pd.read_csv(DATA_PATH)
    text_col = 'review' if 'review' in df.columns else 'text'
    texts = df[text_col].tolist()[:100]
    true_labels = df['label'].tolist()[:100]
    
    hardware_profiles = [
        {"name": "IoT Sensor Node", "cores": 1},
        {"name": "Budget Mobile Phone", "cores": 2},
        {"name": "Mid-Range Smartphone", "cores": 3},
        {"name": "Low-End Laptop", "cores": 4}
    ]
    
    results = []

    for profile in hardware_profiles:
        cores = profile["cores"]
        core_label = "Core" if cores == 1 else "Cores"
        device_name = f"{profile['name']}\n({cores} {core_label})"
        
        print(f"\nEmulating {profile['name']} ({cores} {core_label})...")
        
        if os.path.exists(MODEL_A_DIR):
            print("  Testing Model A (PyTorch)...")
            lat, f1 = benchmark_pytorch_edge(MODEL_A_DIR, texts, true_labels, cores, "distilbert-base-multilingual-cased")
            results.append({"Emulated_Device": device_name, "Model": "Model A (Base DistilBERT)", "Macro_F1": f1, "Latency_ms": lat})
            
        if os.path.exists(MODEL_B_DIR):
            print("  Testing Model B (PyTorch)...")
            lat, f1 = benchmark_pytorch_edge(MODEL_B_DIR, texts, true_labels, cores, "distilbert-base-multilingual-cased")
            results.append({"Emulated_Device": device_name, "Model": "Model B (DAPT-DistilBERT)", "Macro_F1": f1, "Latency_ms": lat})
            
        if os.path.exists(MODEL_C_DIR):
            print("  Testing Model C (PyTorch)...")
            lat, f1 = benchmark_pytorch_edge(MODEL_C_DIR, texts, true_labels, cores, "xlm-roberta-base")
            results.append({"Emulated_Device": device_name, "Model": "Model C (XLM-R Base)", "Macro_F1": f1, "Latency_ms": lat})
        
        if os.path.exists(MODEL_D_DIR):
            print("  Testing Model D (ONNX)...")
            lat, f1 = benchmark_onnx_edge(MODEL_D_DIR, texts, true_labels, cores)
            results.append({"Emulated_Device": device_name, "Model": "Model D (Optimized DAPT)", "Macro_F1": f1, "Latency_ms": lat})

    results_df = pd.DataFrame(results)
    results_df = results_df.round({"Macro_F1": 4, "Latency_ms": 2})

    csv_path = os.path.join(REPORTS_DIR, "edge_hardware_emulation_results.csv")
    
    csv_df = results_df.copy()
    csv_df["Emulated_Device"] = csv_df["Emulated_Device"].str.replace("\n", " ")
    csv_df.to_csv(csv_path, index=False)
    
    print("\nGenerating visual reports...")
    generate_visualizations(results_df)
    
    print(f"\n[SUCCESS] Emulation data saved to: {csv_path}")
    print(f"[SUCCESS] Visual charts saved to: {FIGURES_DIR}")
    print("\n--- FINAL HARDWARE BENCHMARK RESULTS ---")
    print(csv_df.to_string(index=False))

if __name__ == "__main__":
    main()