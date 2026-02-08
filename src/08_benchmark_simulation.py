import time
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

def benchmark_simulation(model_path, tokenizer_path):
    print(f"\n" + "="*50)
    print(f"PERFORMANCE VALIDATION: {os.path.basename(model_path)}")
    print(f"="*50)
    
    # --- FAIRNESS FIX ---
    # Allow the CPU to use its specialized instruction sets (AVX-512)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 0
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    # We remove the single-thread constraint to allow VNNI acceleration
    session = ort.InferenceSession(model_path, options, providers=["CPUExecutionProvider"])
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    text = "Ang ganda ng product na ito sobrang sulit."
    inputs = tokenizer(text, return_tensors="np", padding="max_length", max_length=128, truncation=True)
    
    # Ensure types match the quantized graph requirements
    input_feed = {
        "input_ids": inputs["input_ids"].astype(np.int64), 
        "attention_mask": inputs["attention_mask"].astype(np.int64)
    }

    # Warmup: Crucial for JIT and cache stabilization
    print("Stabilizing hardware cache (Warmup)...")
    for _ in range(50): session.run(None, input_feed)

    # Simulation Loop
    latencies = []
    print("Executing 1000 Inference Trials...")
    for _ in range(1000):
        start = time.perf_counter()
        session.run(None, input_feed)
        latencies.append((time.perf_counter() - start) * 1000)

    avg_latency = np.mean(latencies)
    print(f" Results Captured:")
    print(f"   Average Latency: {avg_latency:.2f} ms")
    print(f"   P95 Latency: {np.percentile(latencies, 95):.2f} ms")
    print(f"   Throughput: {1000 / (sum(latencies)/1000):.2f} samples/sec")

if __name__ == "__main__":
    # Benchmark Model B (The Baseline for Model D)
    # This ensures your 'Speedup Factor' is calculated correctly.
    benchmark_simulation(
        os.path.join(config.MODEL_B_FINETUNED_DIR, "model.onnx"),
        config.MODEL_B_FINETUNED_DIR
    )
    
    # Benchmark Model D (The Optimized Version)
    benchmark_simulation(
        os.path.join(config.MODEL_D_DIR, "model_quantized.onnx"),
        config.MODEL_D_DIR
    )