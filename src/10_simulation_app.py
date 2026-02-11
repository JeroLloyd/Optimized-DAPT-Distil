import streamlit as st
import onnxruntime as ort
import numpy as np
import os
import sys
import time
import pandas as pd
import torch
import matplotlib.pyplot as plt
import platform
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
from scipy.interpolate import make_interp_spline

# --- CONFIG IMPORT ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    import config
except ImportError:
    st.error("CRITICAL ERROR: Could not import 'config.py'. Ensure this script is in the /src folder.")
    st.stop()

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Thesis Benchmark Platform",
    layout="wide"
)

# --- DARK HCI CSS ---
st.markdown("""
<style>
.metric-card { background-color: #1E1E1E; border-radius: 12px; padding: 18px; border: 1px solid #333; }
.training-box { 
    border: 1px solid #444; border-radius: 10px; padding: 15px; background: #0E1117; min-height: 480px;
    transition: all 0.5s ease;
}
.stage-label { font-size: 10px; color: #888; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 5px;}
.active-stage { border: 2px solid #4CAF50 !important; background: #1a2e1a !important; box-shadow: 0 0 15px rgba(76, 175, 80, 0.2); }
.completed-stage { border: 1px solid #4CAF50; opacity: 0.8; }
.data-preview { font-family: monospace; font-size: 11px; background: #000; padding: 10px; border-radius: 5px; color: #00FF00; height: 100px; overflow-y: auto; }
.stMetric { color: white !important; }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# MODEL REGISTRY
# ----------------------------------------------------
MODELS = {
    "Model A": {"name": "Base DistilBERT", "path": config.MODEL_A_DIR, "type": "pytorch"},
    "Model B": {"name": "DAPT-DistilBERT", "path": config.MODEL_B_FINETUNED_DIR, "type": "pytorch"},
    "Model C": {"name": "XLM-RoBERTa", "path": config.MODEL_C_DIR, "type": "pytorch"},
    "Model D": {
        "name": "Optimized ONNX",
        "path": os.path.join(config.MODEL_D_DIR, "model_quantized.onnx"),
        "tokenizer": config.MODEL_D_DIR,
        "type": "onnx"
    }
}

RESULTS_PATH = os.path.join(config.BASE_DIR, "results", "metrics_summary.csv")

# ----------------------------------------------------
# RESOURCE LOADERS
# ----------------------------------------------------
@st.cache_resource
def load_one_model(key):
    model_conf = MODELS[key]
    path, tok_path = model_conf["path"], model_conf.get("tokenizer", model_conf["path"])
    if not os.path.exists(path): return None, None, "error"
    tokenizer = AutoTokenizer.from_pretrained(tok_path)
    if model_conf["type"] == "onnx":
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        model = ort.InferenceSession(path, options, providers=["CPUExecutionProvider"])
        return tokenizer, model, "onnx"
    model = AutoModelForSequenceClassification.from_pretrained(path)
    model.eval(); model.to("cpu")
    return tokenizer, model, "pytorch"

@st.cache_data
def load_thesis_metrics():
    if os.path.exists(RESULTS_PATH):
        df = pd.read_csv(RESULTS_PATH)
        df.columns = [c.strip() for c in df.columns]
        return df
    return None

def compute_research_metrics(df):
    baseline = df.loc[df["Macro_F1"].idxmax()]
    df["Speedup_Factor"] = baseline["Latency_ms"] / df["Latency_ms"]
    df["Performance_Retention"] = (df["Macro_F1"] / baseline["Macro_F1"]) * 100
    df["Efficiency_Score"] = df["Macro_F1"] / df["Latency_ms"]
    
    pareto_flags = []
    for i, row in df.iterrows():
        dominated = False
        for j, other in df.iterrows():
            if (other["Latency_ms"] <= row["Latency_ms"] and other["Macro_F1"] >= row["Macro_F1"] and j != i):
                dominated = True; break
        pareto_flags.append(not dominated)
    df["Pareto_Optimal"] = pareto_flags
    return df

# ----------------------------------------------------
# ACADEMIC PARETO CHART (REAL DATA)
# ----------------------------------------------------
def plot_pareto_with_indicator(df):
    plt.style.use('default') 
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    ax.set_facecolor('white'); fig.patch.set_facecolor('white')
    df_plot = df.copy().sort_values("Latency_ms")
    x, y = df_plot["Latency_ms"].values, df_plot["Macro_F1"].values
    ax.set_xlim(0, 160); ax.set_xticks(range(0, 161, 20))
    ax.set_ylim(0.8140, 0.8360); ax.set_yticks(np.arange(0.8140, 0.8361, 0.002))
    ax.grid(True, linestyle='-', color='#E0E0E0', linewidth=0.6, zorder=0)
    ax.set_xlabel("Inference Latency (ms)", fontsize=13, fontweight='bold')
    ax.set_ylabel("Macro F1 Score", fontsize=13, fontweight='bold')
    ax.axvspan(0, 20, color='#D4EDDA', alpha=0.4, zorder=1)
    ax.text(10, 0.8355, "Real-Time Deployment Zone (<20ms)", ha='center', va='top', fontsize=11, color='#155724', fontweight='bold', zorder=5)

    if len(x) > 2:
        x_smooth = np.linspace(x.min(), x.max(), 300)
        spline = make_interp_spline(x, y, k=2)
        ax.plot(x_smooth, spline(x_smooth), color='#003366', linewidth=3.5, zorder=2)
    else:
        ax.plot(x, y, color='#003366', linewidth=3.5, zorder=2)
    
    for _, row in df_plot.iterrows():
        is_proposed = "Optimized" in row["Model_Name"]
        color, marker = ('red', 'D') if is_proposed else ('#002366', 'o')
        ax.scatter(row["Latency_ms"], row["Macro_F1"], color=color, marker=marker, s=200 if is_proposed else 150, zorder=6, edgecolors='black' if is_proposed else 'white')
        
        if is_proposed: label, xy_off, conn = "PROPOSED MODEL", (25, -40), "arc3,rad=-0.1"
        elif "XLM" in row["Model_Name"]: label, xy_off, conn = "XLM-RoBERTa (Teacher)", (-50, 30), "arc3,rad=0.1"
        elif "DAPT" in row["Model_Name"]: label, xy_off, conn = "DAPT-DistilBERT", (30, 20), "arc3,rad=-0.2"
        else: label, xy_off, conn = "Base DistilBERT", (-60, -35), "arc3,rad=0.2"

        ax.annotate(label, xy=(row["Latency_ms"], row["Macro_F1"]), xytext=xy_off, textcoords='offset points', 
                    fontsize=10, color=color, fontweight='bold', arrowprops=dict(arrowstyle='->', color='black', lw=1.2, connectionstyle=conn), zorder=10)

    ax.set_title("Latency–Accuracy Pareto Frontier for Deployment-Aware NLP", fontsize=16, fontweight='bold', pad=25)
    st.pyplot(fig)

# ----------------------------------------------------
# TRAINING ARCHITECTURE VISUALIZATION
# ----------------------------------------------------
def render_training_simulation():
    st.header("Model Training Architecture & Lifecycle")
    st.caption("Visualizing the side-by-side progression from raw Taglish data to Optimized ONNX graphs.")
    
    # Sample data based on your datasets
    raw_samples = [
        "Ang ganda ng item na to, worth it bilhin!",
        "Medyo matagal ang delivery but okay naman.",
        "Worst experience ever. Scammer ang seller.",
        "Sulit na sulit ang bayad. 5 stars."
    ]
    
    if st.button(" INITIATE PARALLEL TRAINING PIPELINE"):
        cols = st.columns(4)
        placeholders = [cols[i].empty() for i in range(4)]
        
        epochs = 15
        for step in range(epochs):
            progress = (step + 1) / epochs
            
            for i, (key, meta) in enumerate(MODELS.items()):
                # Simulation Data Logic
                base_loss = 0.9 / (step + 1)
                base_acc = 0.5 + (0.32 * progress)
                
                # Model-specific variations
                if key == "Model C": base_acc += 0.02 
                if key in ["Model B", "Model D"]: base_loss *= 0.85 # DAPT converging faster
                
                with placeholders[i].container():
                    st.markdown(f"### {key}")
                    st.caption(meta['name'])
                    
                    # STAGE 1: DATA PIPELINE (Ref: 02_data_cleaning_dapt.py)
                    s1_active = progress < 0.25
                    s1_class = "active-stage" if s1_active else "completed-stage"
                    
                    sample_text = raw_samples[step % 4]
                    if not s1_active: sample_text = sample_text.lower().replace(",", "") # Simple clean simulation
                    
                    st.markdown(f"""<div class='training-box {s1_class}'>
                        <p class='stage-label'>Stage 1: Pre-processing</p>
                        <b>Active Logic:</b> 02_data_cleaning_dapt.py<br>
                        <div class='data-preview'><b>Data Stream:</b><br>{sample_text}</div>
                        <small>Cleaning Taglish nuances...</small>
                    """, unsafe_allow_html=True)
                    
                    # STAGE 2: KNOWLEDGE INTAKE (Ref: 04_train_stage1_dapt.py, 05_train_stage2_finetune.py)
                    s2_active = progress >= 0.25 and progress < 0.85
                    s2_class = "active-stage" if s2_active else ("completed-stage" if progress >= 0.85 else "")
                    
                    dapt_status = " DAPT ACTIVE" if key in ["Model B", "Model D"] else "❌ Standard"
                    trainer = "WeightedTrainer" if key != "Model A" else "DefaultTrainer"
                    
                    st.markdown(f"""<div style='margin-top:10px;'>
                        <p class='stage-label'>Stage 2: Training / DAPT</p>
                        <b>DAPT Status:</b> {dapt_status}<br>
                        <b>Loss:</b> {base_loss + np.random.normal(0, 0.01):.4f}<br>
                        <b>F1-Score:</b> {base_acc + np.random.normal(0, 0.005):.4f}<br>
                        <progress value='{progress}' max='1' style='width:100%'></progress>
                    </div>""", unsafe_allow_html=True)

                    # STAGE 3: HARDWARE OPTIMIZATION (Ref: 06_optimize_stage3_model_d.py)
                    s3_active = progress >= 0.85
                    s3_class = "active-stage" if s3_active and key == "Model D" else ""
                    
                    arch = " AVX-512 VNNI" if key == "Model D" else "N/A (PyTorch FP32)"
                    status_msg = "Optimizing Graph..." if s3_active and key == "Model D" else "Waiting..."
                    if progress >= 0.95 and key == "Model D": status_msg = "Model D Ready (ONNX)"
                    
                    st.markdown(f"""<div style='margin-top:10px;'>
                        <p class='stage-label'>Stage 3: Optimization</p>
                        <b>Architecture:</b> {arch}<br>
                        <b>State:</b> {status_msg}<br>
                    </div></div>""", unsafe_allow_html=True)
                    
            time.sleep(0.8) # Slowed for observation

# ----------------------------------------------------
# SIDEBAR NAVIGATION
# ----------------------------------------------------
with st.sidebar:
    st.header("System Controls")
    mode = st.radio("Select View", ["Research Dashboard", "Live Benchmark", "Training Architecture", "Pipeline Inspector"])
    st.divider()
    st.info(f"**Host Hardware:**\n{platform.processor()}\n\n**Acceleration:**\nAVX-512 VNNI (Model D Only)")

# ----------------------------------------------------
# MAIN ROUTING LOGIC
# ----------------------------------------------------
st.title("Thesis Simulation Platform")

if mode == "Research Dashboard":
    st.header("Optimized-DAPT DistillmBERT Evaluation")
    df_static = load_thesis_metrics()
    if df_static is None:
        st.warning("metrics_summary.csv not found."); st.stop()
    
    df_static = compute_research_metrics(df_static)
    proposed = df_static.loc[df_static["Model_Name"].str.contains("Optimized", case=False)].iloc[0]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Macro F1", f"{proposed['Macro_F1']:.4f}")
    c2.metric("Latency", f"{proposed['Latency_ms']:.2f} ms")
    c3.metric("Model Size", f"{proposed['Model_Size_MB']:.1f} MB")
    c4.metric("Speedup", f"{proposed['Speedup_Factor']:.2f}x")
    c5.metric("Retention", f"{proposed['Performance_Retention']:.2f}%")

    st.subheader("Pareto Frontier — Deployment Perspective")
    plot_pareto_with_indicator(df_static)

elif mode == "Training Architecture":
    render_training_simulation()

elif mode == "Live Benchmark":
    st.header("Real-Time Multi-Run Benchmark")
    text = st.text_area("Input Text (Taglish)", "Sobrang sulit ng item na to, ang bilis pa ng delivery!")
    if st.button("START BENCHMARK"):
        # Stage 1: Scanning
        st.subheader(" Stage 1: Lexical Scanning")
        pipeline_status = st.empty()
        tokenizer, _, _ = load_one_model("Model D")
        tokens = tokenizer.tokenize(text)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        
        words = text.split(); current_view = ""
        for word in words:
            current_view += f" <span style='color:#4CAF50; font-weight:bold;'>{word}</span>"
            pipeline_status.markdown(f"<div style='padding:15px; border-left:5px solid #4CAF50; background:#1a1a1a;'><b>Reading:</b> {current_view}</div>", unsafe_allow_html=True)
            time.sleep(0.5)

        # Stage 2: Vectorization
        st.subheader(" Stage 2: Numerical Vectorization")
        t_cols = st.columns(min(len(tokens), 8))
        for i, t in enumerate(tokens[:8]):
            with t_cols[i]:
                st.markdown(f"<div style='text-align:center;'>{t}<br>↓<br><b style='background:#1a2e1a; padding:2px;'>{token_ids[i]}</b></div>", unsafe_allow_html=True)
                time.sleep(0.4)

        # Stage 3: Inference
        st.subheader(" Stage 3: Multi-Trial Inference")
        progress_bar = st.progress(0); results = []
        for idx, key in enumerate(MODELS.keys()):
            tokenizer, model, engine = load_one_model(key)
            if engine == "error": continue
            def run_inf(t, m, e, txt):
                ins = t(txt, return_tensors="pt" if e == "pytorch" else "np", padding=True, truncation=True, max_length=128)
                if e == "onnx": return m.run(None, {k: v.astype(np.int64) for k, v in ins.items()})[0][0]
                else: 
                    ins = {k: v.to("cpu") for k, v in ins.items()}
                    with torch.no_grad(): return m(**ins).logits[0].numpy()
            
            for _ in range(3): run_inf(tokenizer, model, engine, text) # Warmup
            latencies = []
            for _ in range(15):
                t1 = time.perf_counter()
                logits = run_inf(tokenizer, model, engine, text)
                latencies.append((time.perf_counter() - t1) * 1000)
                time.sleep(0.05)
            
            probs = softmax(logits)
            results.append({"Model": key, "Latency (ms)": np.mean(latencies), "Prediction": ["Negative", "Neutral", "Positive"][np.argmax(probs)], "Confidence": f"{np.max(probs)*100:.1f}%"})
            progress_bar.progress((idx + 1) / 4)
        st.table(pd.DataFrame(results))

elif mode == "Pipeline Inspector":
    st.header("Model Pipeline Inspector")
    selected = st.selectbox("Select Model", list(MODELS.keys()), index=3)
    text = st.text_area("Input Text", "Sulit ito!")
    if st.button("INSPECT"):
        tokenizer, model, engine = load_one_model(selected)
        inputs = tokenizer(text, return_tensors="pt")
        st.write("### Tokens")
        st.code(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]))
        # Inference for Inspector
        ins = tokenizer(text, return_tensors="pt" if engine == "pytorch" else "np", padding=True, truncation=True)
        if engine == "onnx": logits = model.run(None, {k: v.astype(np.int64) for k, v in ins.items()})[0][0]
        else:
            with torch.no_grad(): logits = model(**ins).logits[0].numpy()
        st.bar_chart(pd.DataFrame({"Probability": softmax(logits)}, index=["Negative", "Neutral", "Positive"]))