"""
Edge CPU Hardware Emulation and Benchmarking Script.

This module simulates the performance of various sequence classification models
on resource-constrained edge devices. It enforces strict CPU thread limits using
both PyTorch and ONNX Runtime to evaluate inference latency and efficiency.
The script also generates visualization plots for latency and Pareto efficiency.
"""

import os
import time
import gc
import warnings

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import onnxruntime as ort
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from optimum.onnxruntime import ORTModelForSequenceClassification

warnings.filterwarnings("ignore", category=FutureWarning)

# ==============================================================================
# AGGRESSIVE COMPATIBILITY PATCHES
# ==============================================================================
import transformers
import transformers.models.auto

class MockAutoModelForVision2Seq:
    """Mock class injected to bypass missing Vision2Seq dependencies."""
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        raise NotImplementedError("Mock class for compatibility.")

setattr(transformers, "AutoModelForVision2Seq", MockAutoModelForVision2Seq)
setattr(transformers.models.auto, "AutoModelForVision2Seq", MockAutoModelForVision2Seq)

# Disable offline mode checks to ensure remote tokenizer loading works
if not hasattr(transformers.utils, 'is_offline_mode'):
    transformers.utils.is_offline_mode = lambda: False

# ==============================================================================
# PATH CONFIGURATION
# ==============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(SCRIPT_DIR)
BASE_DIR = os.path.dirname(SRC_DIR)

DATA_PATH = os.path.join(BASE_DIR, 'data', '03_processed', 'FiReCS_Final', 'test.csv')
FINAL_METRICS_PATH = os.path.join(BASE_DIR, 'reports', 'metrics', 'final_metrics.csv')

# SYNCED: Model directories mapping to earlier training stages
MODEL_A_DIR = os.path.join(BASE_DIR, 'models', 'model_a_base')
MODEL_B_DIR = os.path.join(BASE_DIR, 'models', 'model_b_dapt')
MODEL_C_DIR = os.path.join(BASE_DIR, 'models', 'model_c_xlmr')
MODEL_D_DIR = os.path.join(BASE_DIR, 'models', 'model_d_onnx')

FIGURES_DIR = os.path.join(BASE_DIR, 'reports', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

MAX_LATENCY_SAMPLES = 200


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def clear_memory():
    """Forces garbage collection and clears the CUDA memory cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def safe_load_tokenizer(model_path, fallback_name):
    """
    Safely loads a tokenizer from a path, falling back to a default base model.

    Args:
        model_path (str): The local directory containing tokenizer files.
        fallback_name (str): The Hugging Face identifier for the fallback tokenizer.

    Returns:
        PreTrainedTokenizer: The initialized Hugging Face tokenizer.
    """
    try:
        return AutoTokenizer.from_pretrained(model_path)
    except Exception:
        return AutoTokenizer.from_pretrained(fallback_name)


def get_official_f1(model_name):
    """
    Fetches the official Macro F1 score from Stage 8 metrics for consistent reporting.

    Args:
        model_name (str): The exact name of the model to query.

    Returns:
        float: The Macro F1 score, or 0.0 if not found.
    """
    try:
        if os.path.exists(FINAL_METRICS_PATH):
            df = pd.read_csv(FINAL_METRICS_PATH)
            # SYNCED: Matches nomenclature from Script 06 logs
            row = df[df['Model Name'] == model_name]
            if not row.empty:
                return float(row.iloc[0]['Macro F1 Score'])
    except Exception as e:
        print(f"  [WARN] Could not sync official metrics for {model_name}: {e}")
    return 0.0


# ==============================================================================
# CORE LATENCY MEASUREMENT
# ==============================================================================
def measure_latency(model_path, texts, cores, fallback_tokenizer, is_onnx=False):
    """
    Profiles hardware latency by forcing models to execute within specific core limits.

    Args:
        model_path (str): The directory containing the model weights.
        texts (list): A list of text samples to run inference on.
        cores (int): The number of CPU threads/cores to restrict execution to.
        fallback_tokenizer (str): The fallback tokenizer string.
        is_onnx (bool): Flag indicating if the model requires the ONNX runtime.

    Returns:
        float: The average inference latency in milliseconds.
    """
    tokenizer = safe_load_tokenizer(model_path, fallback_tokenizer)
    latencies = []
    
    # Sub-sample texts for faster, consistent evaluation
    np.random.seed(42)
    sample_indices = np.random.choice(len(texts), min(MAX_LATENCY_SAMPLES, len(texts)), replace=False)
    sampled_texts = [texts[i] for i in sample_indices]
    
    if is_onnx:
        # EMULATION: Lock ONNX Runtime to the specified thread count
        options = ort.SessionOptions()
        options.intra_op_num_threads = cores
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        model = ORTModelForSequenceClassification.from_pretrained(
            model_path, provider="CPUExecutionProvider", session_options=options
        )
        
        # Warmup phase
        warmup = tokenizer(sampled_texts[0], return_tensors="pt", truncation=True, padding='max_length', max_length=128)
        if "distilbert" in fallback_tokenizer.lower(): 
            warmup.pop("token_type_ids", None)
        _ = model(**warmup)

        # Timed inference loop
        for text in sampled_texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding='max_length', max_length=128)
            if "distilbert" in fallback_tokenizer.lower(): 
                inputs.pop("token_type_ids", None)
            
            start = time.perf_counter()
            _ = model(**inputs)
            latencies.append((time.perf_counter() - start) * 1000)
    else:
        # EMULATION: Lock PyTorch execution to the specified thread count
        torch.set_num_threads(cores)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.to("cpu")
        model.eval()
        
        # Warmup phase
        warmup = tokenizer(sampled_texts[0], return_tensors="pt", truncation=True, padding='max_length', max_length=128)
        if "distilbert" in fallback_tokenizer.lower(): 
            warmup.pop("token_type_ids", None)
        with torch.no_grad(): 
            _ = model(**warmup)

        # Timed inference loop
        for text in sampled_texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding='max_length', max_length=128)
            if "distilbert" in fallback_tokenizer.lower(): 
                inputs.pop("token_type_ids", None)
            
            start = time.perf_counter()
            with torch.no_grad(): 
                _ = model(**inputs)
            latencies.append((time.perf_counter() - start) * 1000)
            
    return np.mean(latencies)


# ==============================================================================
# VISUALIZATION LOGIC
# ==============================================================================
def generate_visualizations(df):
    """
    Generates and saves performance charts based on the emulation results.

    Args:
        df (pd.DataFrame): The dataframe containing the latency and F1 metrics.
    """
    EDGE_DIR = os.path.join(FIGURES_DIR, "08_edge_hardware_emulation_results")
    os.makedirs(EDGE_DIR, exist_ok=True)

    sns.set_theme(style="whitegrid", rc={"axes.facecolor": "#f8f9fa"})
    palette = ["#7f8c8d", "#3498db", "#e74c3c", "#2ecc71"]
    
    # --------------------------------------------------------------------------
    # FIG 8a: Latency Bar Chart
    # --------------------------------------------------------------------------
    plt.figure(figsize=(14, 7))
    sns.barplot(data=df, x='Emulated_Device', y='Latency_ms', hue='Model', palette=palette)
    plt.title("Inference Latency Across Edge Devices", fontsize=16, fontweight='bold', pad=15)
    plt.ylabel("Average Latency (ms)", fontsize=12, fontweight='bold')
    plt.xlabel("Emulated Hardware Profile", fontsize=12, fontweight='bold')
    plt.legend(title=None, bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(EDGE_DIR, "8a_edge_latency.png"), dpi=300)
    plt.close()

    # --------------------------------------------------------------------------
    # FIG 8b: Efficiency Relplot (Pareto)
    # --------------------------------------------------------------------------
    g = sns.relplot(
        data=df, x='Latency_ms', y='Macro_F1', hue='Model', col='Emulated_Device',
        col_wrap=2, kind='scatter', s=300, palette=palette, height=4.5, aspect=1.3,
        edgecolor='black', alpha=0.8
    )
    
    g.fig.suptitle("Edge Efficiency: Latency vs. Accuracy", fontsize=16, fontweight='bold', y=1.05)
    g.set_axis_labels("Inference Latency (ms) [Lower is Better]", "Macro F1 Score [Higher is Better]", fontweight='bold')
    g.set_titles(col_template="{col_name}", size=13, weight='bold')
    sns.move_legend(g, "upper center", bbox_to_anchor=(0.5, -0.05), ncol=4, title=None, frameon=True)
    plt.savefig(os.path.join(EDGE_DIR, "8b_edge_efficiency.png"), dpi=300, bbox_inches='tight')
    plt.close()


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def main():
    """
    Coordinates the hardware emulation benchmark.
    
    This function establishes specific edge hardware profiles, retrieves official 
    F1 scores, and measures model latency under constrained threaded environments.
    It exports the findings as both a structured CSV and visualizations.
    """
    print("=== EDGE HARDWARE EMULATION BENCHMARK ===")
    
    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] Test data not found: {DATA_PATH}")
        return
        
    df_test = pd.read_csv(DATA_PATH)
    text_col = 'review' if 'review' in df_test.columns else 'text'
    texts = df_test[text_col].tolist()
    
    # Define simulated hardware constraints based on CPU cores
    hardware_profiles = [
        {"name": "IoT Sensor Node", "cores": 1},
        {"name": "Budget Mobile Phone", "cores": 2},
        {"name": "Mid-Range Smartphone", "cores": 3},
        {"name": "Low-End Laptop", "cores": 4}
    ]
    
    # SYNCED: Nomenclature matches Stage 2 logs
    models_to_test = [
        ("Model A (Base DistilmBERT)", MODEL_A_DIR, "distilbert-base-multilingual-cased", False),
        ("Model B (DAPT-DistilmBERT)", MODEL_B_DIR, "distilbert-base-multilingual-cased", False),
        ("Model C (XLM-R Base)", MODEL_C_DIR, "xlm-roberta-base", False),
        ("Model D (Optimized DAPT)", MODEL_D_DIR, "distilbert-base-multilingual-cased", True)
    ]

    results = []
    dynamic_f1_cache = {}

    # --------------------------------------------------------------------------
    # Phase 1: Sync Official Metrics
    # --------------------------------------------------------------------------
    print("Phase 1: Syncing official F1 scores from Stage 8...")
    for name, path, tokenizer_name, is_onnx in models_to_test:
        f1 = get_official_f1(name)
        dynamic_f1_cache[name] = f1
        print(f"  Synced {name}: {f1:.4f}")

    # --------------------------------------------------------------------------
    # Phase 2: Hardware Constraint Profiling
    # --------------------------------------------------------------------------
    print("\nPhase 2: Profiling hardware latency constraints...")
    for profile in hardware_profiles:
        cores = profile["cores"]
        device_name = f"{profile['name']}\n({cores} Cores)"
        print(f"\nEmulating {profile['name']} ({cores} Cores)...")
        
        for name, path, tokenizer_name, is_onnx in models_to_test:
            if os.path.exists(path):
                print(f"  Profiling {name}...")
                lat = measure_latency(path, texts, cores, tokenizer_name, is_onnx)
                
                results.append({
                    "Emulated_Device": device_name,
                    "Model": name,
                    "Macro_F1": dynamic_f1_cache[name],
                    "Latency_ms": lat
                })
                clear_memory()

    # --------------------------------------------------------------------------
    # Phase 3: Export and Reporting
    # --------------------------------------------------------------------------
    results_df = pd.DataFrame(results)
    results_df["Latency_ms"] = results_df["Latency_ms"].round(2)

    EDGE_DIR = os.path.join(FIGURES_DIR, "08_edge_hardware_emulation_results")
    os.makedirs(EDGE_DIR, exist_ok=True)
    csv_path = os.path.join(EDGE_DIR, "edge_hardware_emulation_results.csv")
    
    # Save a clean CSV without newline characters in the device name
    csv_df = results_df.copy()
    csv_df["Emulated_Device"] = csv_df["Emulated_Device"].str.replace("\n", " ")
    csv_df.to_csv(csv_path, index=False)
    
    print("\nGenerating visual reports for Chapter 4...")
    generate_visualizations(results_df)
    
    print(f"\n[SUCCESS] Emulation data saved to: {csv_path}")
    print(csv_df.to_string(index=False))


if __name__ == "__main__":
    main()