import streamlit as st
import onnxruntime as ort
import numpy as np
import os
import time
import pandas as pd
import torch
import matplotlib.pyplot as plt
import platform
import altair as alt
import transformers
import transformers.models.auto
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import re

# --- NEW PREPROCESSING ALGORITHM ---
def apply_thesis_preprocessing(text):
    """Replicates the text cleaning and gibberish salvaging from script 04."""
    # 1. Basic Clean
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 2. Gibberish Filtering
    words = text.split()
    clean_words = []
    for word in words:
        if len(word) > 15: continue
        vowels = len(re.findall(r'[aeiou]', word))
        if len(word) > 0:
            ratio = vowels / len(word)
            if ratio < 0.2 or ratio > 0.9: continue
        clean_words.append(word)
        
    return " ".join(clean_words) if len(clean_words) >= 3 else None

# Strict Inter-op threading limit for accurate Edge CPU Emulation
try:
    torch.set_num_interop_threads(1)
except Exception:
    pass

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
SRC_DIR = os.path.dirname(SCRIPT_DIR)
BASE_DIR = os.path.dirname(SRC_DIR)

MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'reports', 'metrics')
RESULTS_PATH = os.path.join(RESULTS_DIR, "final_metrics.csv")

# --- PAGE CONFIG ---
st.set_page_config(page_title="Evaluation Framework", layout="wide", page_icon="⚡")

st.markdown("""
<style>
    .main { background-color: #0E1117; }
    .metric-card { background-color: #1E1E1E; border-radius: 12px; padding: 18px; border: 1px solid #333; margin-bottom: 10px; }
    h1, h2, h3 { color: #e6e6e6 !important; }
    .stMetric > div { background-color: #1E1E1E !important; padding: 10px; border-radius: 5px; border: 1px solid #333; }
    .stMetric label { color: #aaaaaa !important; }
    .stMetric div[data-testid="stMetricValue"] { color: #ffffff !important; }
</style>
""", unsafe_allow_html=True)

# --- MODEL REGISTRY ---
MODELS = {
    "Model A": {"name": "Base DistilmBERT", "path": os.path.join(MODELS_DIR, "model_a_base"), "type": "pytorch"},
    "Model B": {"name": "DAPT-DistilmBERT", "path": os.path.join(MODELS_DIR, "model_b_dapt"), "type": "pytorch"},
    "Model C": {"name": "XLM-RoBERTa", "path": os.path.join(MODELS_DIR, "model_c_xlmr"), "type": "pytorch"},
    "Model D": {"name": "Optimized ONNX", "path": os.path.join(MODELS_DIR, "model_d_onnx"), "tokenizer": os.path.join(MODELS_DIR, "model_d_onnx"), "type": "onnx"}
}

# --- RESOURCE LOADERS ---
@st.cache_resource
def load_one_model(key, device="cpu"):
    try:
        model_conf = MODELS[key]
        path = model_conf["path"]
        tok_path = model_conf.get("tokenizer", model_conf["path"])
        
        if not os.path.exists(path) and not os.path.exists(tok_path):
             return None, None, "error"

        try:
            tokenizer = AutoTokenizer.from_pretrained(tok_path)
        except Exception:
            if "Model C" in key:
                tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
            else:
                tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")

        if model_conf["type"] == "onnx":
            options = ort.SessionOptions()
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
            model_file = os.path.join(path, "model.onnx") if os.path.isdir(path) else path
            
            try:
                model = ort.InferenceSession(model_file, options, providers=providers)
            except Exception:
                model = ort.InferenceSession(model_file, options, providers=["CPUExecutionProvider"])
                
            return tokenizer, model, "onnx"
        else:
            model = AutoModelForSequenceClassification.from_pretrained(path)
            model.to(device)
            model.eval()
            return tokenizer, model, "pytorch"
            
    except Exception as e:
        print(f"Error loading {key}: {e}")
        return None, None, "error"

def get_edge_onnx_session(path, cores):
    """Creates a temporary, non-cached ONNX session with strictly limited threads."""
    options = ort.SessionOptions()
    options.intra_op_num_threads = cores
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    model_file = os.path.join(path, "model.onnx") if os.path.isdir(path) else path
    return ort.InferenceSession(model_file, options, providers=["CPUExecutionProvider"])

@st.cache_data
def load_thesis_metrics():
    if not os.path.exists(RESULTS_PATH):
        return None
        
    df = pd.read_csv(RESULTS_PATH)
    df.columns = [c.strip() for c in df.columns] 
    
    return df
    

def compute_research_metrics(df):
    # Dynamically find the correct latency column based on what Script 08 exported
    if "Empirical Time Complexity (Latency ms)" in df.columns:
        latency_col = "Empirical Time Complexity (Latency ms)"
    elif "Avg Latency (Overall) ms" in df.columns:
        latency_col = "Avg Latency (Overall) ms"
    else:
        latency_col = "Avg Latency (ms)" # Fallback
    
    # Calculate Speedup Factor
    if not df.empty:
        baseline = df.loc[df[latency_col].idxmax()]
        df["Speedup_Factor"] = baseline[latency_col] / df[latency_col]
    else:
        df["Speedup_Factor"] = 1.0

    best_f1 = df["Macro F1 Score"].max()
    df["Performance_Retention"] = (df["Macro F1 Score"] / best_f1) * 100
    
    # --- NEW EFFICIENCY METRICS ---
    # Compute Efficiency: F1 score achieved per 1000ms of latency
    df["Compute_Efficiency"] = (df["Macro F1 Score"] / df[latency_col]) * 1000
    
    # Storage Efficiency: F1 score achieved per MB of disk space
    if "Empirical Space Complexity (Storage MB)" in df.columns:
        df["Storage_Efficiency"] = (df["Macro F1 Score"] / df["Empirical Space Complexity (Storage MB)"]) * 1000
    elif "Model Size (MB)" in df.columns:
        df["Storage_Efficiency"] = (df["Macro F1 Score"] / df["Model Size (MB)"]) * 1000
    elif "Storage_MB" in df.columns:
        df["Storage_Efficiency"] = (df["Macro F1 Score"] / df["Storage_MB"]) * 1000
    else:
        df["Storage_Efficiency"] = 0.0 # Fallback if missing
        
    df = df.rename(columns={latency_col: "Latency_ms", "Macro F1 Score": "Macro_F1", "Model Name": "Model_Name"})
    return df


def plot_horizontal_metric(df, metric_col, title, format_str, highlight_color):
    domain = [
        "Model A (Base DistilmBERT)", 
        "Model B (DAPT-DistilmBERT)", 
        "Model C (XLM-R Base)", 
        "Model D (Optimized DAPT)"
    ]
    # Models A and C = Grey. Models B and D = Green.
    range_colors = ['#34495e', '#34495e', '#34495e', '#2ecc71']

    base = alt.Chart(df).encode(
        y=alt.Y('Model_Name:N', sort=None, title=None, axis=alt.Axis(labelLimit=300, labelColor='white')),
        x=alt.X(f'{metric_col}:Q', title=None, axis=alt.Axis(grid=True, format=format_str, labelColor='white')),
        tooltip=['Model_Name', alt.Tooltip(f'{metric_col}:Q', format=format_str)]
    )
    
    bars = base.mark_bar(cornerRadiusEnd=4, height=35).encode(
        color=alt.Color('Model_Name:N', scale=alt.Scale(domain=domain, range=range_colors), legend=None)
    )
    
    text = base.mark_text(align='left', baseline='middle', dx=5, color='white', fontWeight='bold').encode(
        text=alt.Text(f'{metric_col}:Q', format=format_str)
    )
    
    return (bars + text).properties(title=alt.TitleParams(text=title, color='white'), height=220)

# --- INFERENCE ENGINE ---
def run_inference(txt, tk_obj, mdl_obj, eng_type, key_n, c_dict, dev):
    if eng_type == "onnx":
        inputs = tk_obj(txt, return_tensors="np", padding=True, truncation=True, max_length=128)
        if "token_type_ids" in inputs and ("distilmbert" in c_dict["name"].lower() or key_n == "Model D"):
            del inputs["token_type_ids"]
        inputs = {k: v.astype(np.int64) for k, v in inputs.items()}
        start = time.perf_counter()
        logits = mdl_obj.run(None, inputs)[0][0]
        dur = (time.perf_counter() - start) * 1000
        return logits, dur
    else:
        inputs = tk_obj(txt, return_tensors="pt", padding=True, truncation=True, max_length=128)
        if "token_type_ids" in inputs and ("distilmbert" in c_dict["name"].lower() or "model_a" in key_n.lower() or "model_b" in key_n.lower()):
            del inputs["token_type_ids"]
        inputs = {k: v.to(dev) for k, v in inputs.items()}
        start = time.perf_counter()
        with torch.no_grad():
            logits = mdl_obj(**inputs).logits[0].cpu().numpy()
        dur = (time.perf_counter() - start) * 1000
        return logits, dur

# --- SESSION STATE INITIALIZATION ---
if 'edge_state' not in st.session_state: st.session_state.edge_state = None
if 'diag_state' not in st.session_state: st.session_state.diag_state = None
if 'batch_state' not in st.session_state: st.session_state.batch_state = None

# --- UI LOGIC ---
with st.sidebar:
    st.header("Evaluation Controls")
    mode = st.radio("Select View", ["Evaluation Dashboard", "Edge Emulation", "Diagnostic Inference", "Batch Simulation"])
    st.divider()
    hw_info = platform.processor()
    compute_eng = "CPU Execution Provider"
    if torch.cuda.is_available():
        hw_info += f"\n\n**GPU Profile:**\n{torch.cuda.get_device_name(0)}"
        compute_eng += "\nCUDA Execution Provider"
    st.info(f"**Hardware Profile:**\n{hw_info}\n\n**Compute Engines Available:**\n{compute_eng}")

st.title("Sentiment Analysis Evaluation Framework")

if mode == "Evaluation Dashboard":
    st.header("Comparative Model Evaluation")
    df_static = load_thesis_metrics()
    
    if df_static is None:
        st.error("Metrics file missing. Execute quantitative benchmarking script first.")
    else:
        df_plot = df_static.drop_duplicates(subset=['Model Name']).copy()
        
        df_plot = compute_research_metrics(df_plot)
        
        st.subheader("Key Performance Indicators by Architecture")
        c1, c2 = st.columns(2)
        with c1:
            st.altair_chart(plot_horizontal_metric(df_plot, "Macro_F1", "Macro F1 Score (Higher is Better)", ".4f", "#2ecc71"), use_container_width=True)
            st.altair_chart(plot_horizontal_metric(df_plot, "Speedup_Factor", "Speedup Factor (Higher is Better)", ".2f", "#9b59b6"), use_container_width=True)
        with c2:
            st.altair_chart(plot_horizontal_metric(df_plot, "Latency_ms", "Inference Latency in ms (Lower is Better)", ".2f", "#e74c3c"), use_container_width=True)
            st.altair_chart(plot_horizontal_metric(df_plot, "Performance_Retention", "Performance Retained %", ".1f", "#3498db"), use_container_width=True)
        
        st.divider()
        st.subheader("Efficiency Distribution")
        
        c3, c4 = st.columns(2)
        with c3:
            st.altair_chart(plot_horizontal_metric(df_plot, "Compute_Efficiency", "Compute Efficiency: F1 per 1000ms (Higher is Better)", ".2f", "#f39c12"), use_container_width=True)
        with c4:
            if "Storage_Efficiency" in df_plot.columns:
                st.altair_chart(plot_horizontal_metric(df_plot, "Storage_Efficiency", "Storage Efficiency: F1 per MB (Higher is Better)", ".2f", "#e67e22"), use_container_width=True)
            else:
                st.info("Storage Efficiency data not available.")
      
elif mode == "Edge Emulation":
    st.header("Live Edge Hardware Emulation")
    st.markdown("Enter code-switched text to dynamically benchmark it across 1 to 4 restricted CPU cores. This emulates inference limits for common low-end devices.")
    text_input_edge = st.text_area("Input Code-Switched Text", "Ang ganda ng quality, sulit na sulit ang bayad! Mabilis pa shipping.", key="edge_input")
    
    if st.button("Simulate Hardware Processing"):
        st.session_state.edge_state = {}
        
        # Methodological Consistency: Preprocess before tokenization
        text_to_process = apply_thesis_preprocessing(text_input_edge)
        if not text_to_process:
            st.error("Input rejected: Preprocessing filtered this text as gibberish or insufficient.")
            
        st.session_state.edge_state['cleaned_text'] = text_to_process
        tokenizer, _, _ = load_one_model("Model D", device="cpu")
        if tokenizer:
            tokens = tokenizer.tokenize(text_input_edge)
            token_ids = tokenizer.convert_tokens_to_ids(tokens)
            st.session_state.edge_state['tokens'] = list(zip(tokens, token_ids))
            
        edge_profiles = [
            {"name": "IoT Sensor Node", "cores": 1},
            {"name": "Budget Mobile Phone", "cores": 2},
            {"name": "Mid-Range Smartphone", "cores": 3},
            {"name": "Low-End Laptop", "cores": 4}
        ]
        
        results_data_edge = []
        progress_bar = st.progress(0)
        total_operations = len(edge_profiles) * len(MODELS)
        current_op = 0
        
        with st.spinner("Emulating isolated hardware constraints..."):
            for profile in edge_profiles:
                cores = profile["cores"]
                device_name = f"{profile['name']} ({cores} Cores)"
                
                torch.set_num_threads(cores)
                
                for key, conf in MODELS.items():
                    if conf["type"] == "onnx":
                        tokenizer, _, _ = load_one_model(key, "cpu")
                        if tokenizer is None:
                            current_op += 1; continue
                        try:
                            model = get_edge_onnx_session(conf["path"], cores)
                        except Exception:
                            current_op += 1; continue
                        engine = "onnx"
                    else:
                        tokenizer, model, engine = load_one_model(key, "cpu")
                        if model is None:
                            current_op += 1; continue
                            
                    try:
                        run_inference("warmup", tokenizer, model, engine, key, conf, "cpu")
                    except Exception:
                        pass
                        
                    latencies = []
                    final_logits = None
                    for _ in range(10): 
                        logits, dur = run_inference(text_input_edge, tokenizer, model, engine, key, conf, "cpu")
                        latencies.append(dur)
                        final_logits = logits
                        
                    avg_lat = np.mean(latencies)
                    probs = softmax(final_logits)
                    pred_idx = np.argmax(probs)
                    labels = ["Negative", "Neutral", "Positive"]
                    
                    results_data_edge.append({
                        "Model": key,
                        "Hardware Profile": device_name,
                        "Avg Latency": avg_lat,
                        "Prediction": labels[pred_idx],
                        "Confidence": max(probs),
                        "Probs": probs
                    })
                    
                    current_op += 1
                    progress_bar.progress(current_op / total_operations)
                    
        st.session_state.edge_state['df'] = pd.DataFrame(results_data_edge)

    # --- Render from Memory if Exists ---
    if st.session_state.edge_state is not None:
        st.subheader("1. Subword Tokenization")
        if 'tokens' in st.session_state.edge_state:
            html_tokens = ""
            for t, tid in st.session_state.edge_state['tokens']:
                html_tokens += f"<div style='display:inline-block; margin:2px; padding:5px; background:#1E1E1E; border: 1px solid #333; border-radius:4px; text-align:center;'><span style='color:#2ecc71; font-size:13px; font-weight:bold;'>{t}</span><br><span style='color:#aaaaaa; font-size:11px;'>{tid}</span></div>"
            st.markdown(html_tokens, unsafe_allow_html=True)
        
        st.subheader("2. Computational Execution (Restricted Cores)")
        st.success("Hardware emulation completed successfully.")
        
        st.subheader("3. Live Hardware Latency Comparison")
        df_results_edge = st.session_state.edge_state['df']
        
        st.markdown("**Latency Matrix (ms)**")
        pivot_df = df_results_edge.pivot(index="Model", columns="Hardware Profile", values="Avg Latency")
        
        display_pivot = pivot_df.copy()
        for col in display_pivot.columns:
            display_pivot[col] = display_pivot[col].apply(lambda x: f"{x:.2f} ms")
            
        st.dataframe(display_pivot, use_container_width=True)
        
        st.markdown("**Visual Hardware Benchmark**")
        domain = ["Model A", "Model B", "Model C", "Model D"]
        range_colors = ['#34495e', '#34495e', '#34495e', '#2ecc71']

        live_lat_chart = alt.Chart(df_results_edge).mark_bar(cornerRadiusEnd=3, height=18).encode(
            y=alt.Y('Hardware Profile:N', title=None, sort=None, axis=alt.Axis(labelColor='white', labelFontSize=12, labelFontWeight='bold')),
            yOffset='Model:N',
            x=alt.X('Avg Latency:Q', title='Inference Latency in ms (Lower is Better)', axis=alt.Axis(labelColor='white', titleColor='white', grid=True)),
            color=alt.Color('Model:N', scale=alt.Scale(domain=domain, range=range_colors), legend=alt.Legend(title=None, labelColor='white', orient='top', padding=10)),
            tooltip=['Model', 'Hardware Profile', alt.Tooltip('Avg Latency:Q', format='.2f')]
        ).properties(height=450)
        
        text_lat_edge = alt.Chart(df_results_edge).mark_text(
            align='left', baseline='middle', dx=4, color='white', fontWeight='bold', fontSize=11
        ).encode(
            y=alt.Y('Hardware Profile:N', sort=None),
            yOffset='Model:N',
            x=alt.X('Avg Latency:Q'),
            text=alt.Text('Avg Latency:Q', format='.1f')
        )
        
        st.altair_chart(live_lat_chart + text_lat_edge, use_container_width=True)

elif mode == "Diagnostic Inference":
    st.header("Latency and Classification Assessment (CPU vs GPU)")
    text_input = st.text_area("Input Code-Switched Text", "Ang ganda ng quality, sulit na sulit ang bayad! Mabilis pa shipping.", key="diag_input")
    
    if st.button("Execute Inference Sequence"):
        st.session_state.diag_state = {}
        tokenizer, _, _ = load_one_model("Model D", device="cpu") 
        if tokenizer:
            tokens = tokenizer.tokenize(text_input)
            token_ids = tokenizer.convert_tokens_to_ids(tokens)
            st.session_state.diag_state['tokens'] = list(zip(tokens, token_ids))
        
        results_data = []
        progress_bar = st.progress(0)
        devices_to_test = ["cpu"]
        if torch.cuda.is_available():
            devices_to_test.append("cuda")
            
        total_operations = len(MODELS) * len(devices_to_test)
        current_op = 0
        
        with st.spinner("Measuring response times across architectures and hardware..."):
            for device in devices_to_test:
                for idx, (key, conf) in enumerate(MODELS.items()):
                    tokenizer, model, engine = load_one_model(key, device=device)
                    if model is None:
                        current_op += 1
                        continue
                        
                    try:
                        run_inference("warmup", tokenizer, model, engine, key, conf, device)
                    except Exception:
                        pass 
                    
                    latencies = []
                    final_logits = None
                    for _ in range(20): 
                        logits, dur = run_inference(text_input, tokenizer, model, engine, key, conf, device)
                        latencies.append(dur)
                        final_logits = logits
                        
                    avg_lat = np.mean(latencies)
                    probs = softmax(final_logits)
                    pred_idx = np.argmax(probs)
                    labels = ["Negative", "Neutral", "Positive"]
                    
                    hw_label = "GPU" if device == "cuda" else "CPU"
                    
                    results_data.append({
                        "Model": key,
                        "Hardware": hw_label,
                        "Engine": engine.upper(),
                        "Avg Latency": avg_lat, 
                        "Prediction": labels[pred_idx],
                        "Confidence": probs[pred_idx],
                        "Probs": probs
                    })
                    current_op += 1
                    progress_bar.progress(current_op / total_operations)
                    
        st.session_state.diag_state['df'] = pd.DataFrame(results_data)
        st.session_state.diag_state['results_data'] = results_data

    # --- Render from Memory if Exists ---
    if st.session_state.diag_state is not None:
        st.subheader("1. Subword Tokenization")
        if 'tokens' in st.session_state.diag_state:
            html_tokens = ""
            for t, tid in st.session_state.diag_state['tokens']:
                html_tokens += f"<div style='display:inline-block; margin:2px; padding:5px; background:#1E1E1E; border: 1px solid #333; border-radius:4px; text-align:center;'><span style='color:#2ecc71; font-size:13px; font-weight:bold;'>{t}</span><br><span style='color:#aaaaaa; font-size:11px;'>{tid}</span></div>"
            st.markdown(html_tokens, unsafe_allow_html=True)

        st.subheader("2. Computational Execution (Hardware Benchmark)")
        st.success("Cross-hardware benchmark completed successfully.")

        st.subheader("3. Latency Benchmarks and Confidence Distributions")
        df_results = st.session_state.diag_state['df']
        results_data = st.session_state.diag_state['results_data']
        
        st.markdown("**Latency Matrix (ms)**")
        pivot_df = df_results.pivot(index="Model", columns="Hardware", values="Avg Latency")
        
        display_pivot = pivot_df.copy()
        for col in display_pivot.columns:
            display_pivot[col] = display_pivot[col].apply(lambda x: f"{x:.2f} ms")
        st.dataframe(display_pivot, use_container_width=True)
        
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown("**Visual Hardware Comparison**")
            domain = ["Model A", "Model B", "Model C", "Model D"]
            range_colors = ['#34495e', '#2ecc71', '#34495e', '#2ecc71']
            
            bars_lat = alt.Chart(df_results).mark_bar(cornerRadiusEnd=3, height=18).encode(
                y=alt.Y('Hardware:N', title=None, sort=None, axis=alt.Axis(labelColor='white', labelFontSize=12, labelFontWeight='bold')),
                yOffset='Model:N',
                x=alt.X('Avg Latency:Q', title='Latency (ms)', axis=alt.Axis(labelColor='white', titleColor='white', grid=True)),
                color=alt.Color('Model:N', scale=alt.Scale(domain=domain, range=range_colors), legend=alt.Legend(title=None, labelColor='white', orient='top', padding=10)),
                tooltip=['Model', 'Hardware', alt.Tooltip('Avg Latency:Q', format='.2f')]
            )
            
            text_lat = alt.Chart(df_results).mark_text(
                align='left', baseline='middle', dx=4, color='white', fontWeight='bold', fontSize=11
            ).encode(
                y=alt.Y('Hardware:N', sort=None),
                yOffset='Model:N',
                x=alt.X('Avg Latency:Q'),
                text=alt.Text('Avg Latency:Q', format='.1f')
            )
            
            chart_lat = (bars_lat + text_lat).properties(height=350)
            st.altair_chart(chart_lat, width="stretch")
        
        with c2:
            st.markdown("**Prediction Confidence (CPU Runtime)**")
            all_probs = []
            cpu_results = [res for res in results_data if res["Hardware"] == "CPU"]
            for res in cpu_results:
                if res["Probs"] is not None:
                    # 0=Negative, 1=Neutral, 2=Positive mapping
                    all_probs.append({"Model": res["Model"], "Sentiment": "Negative", "Probability": float(res["Probs"][0])})
                    all_probs.append({"Model": res["Model"], "Sentiment": "Neutral", "Probability": float(res["Probs"][1])})
                    all_probs.append({"Model": res["Model"], "Sentiment": "Positive", "Probability": float(res["Probs"][2])})
            
            if all_probs:
                df_probs = pd.DataFrame(all_probs)
                sentiment_order = ['Negative', 'Neutral', 'Positive']
                
                # Grouped Horizontal Bar Chart using yOffset for high readability
                chart_conf = alt.Chart(df_probs).mark_bar(cornerRadiusEnd=3, height=18).encode(
                    y=alt.Y('Model:N', title=None, axis=alt.Axis(labelColor='white', labelFontSize=12, labelFontWeight='bold')),
                    yOffset='Sentiment:N', # This creates the side-by-side grouping
                    x=alt.X('Probability:Q', title='Confidence level', axis=alt.Axis(format='%', labelColor='white', titleColor='white')),
                    color=alt.Color('Sentiment:N', 
                        scale=alt.Scale(domain=sentiment_order, range=['#e74c3c', '#95a5a6', '#2ecc71']), 
                        sort=sentiment_order,
                        legend=alt.Legend(title=None, labelColor='white', orient='top')
                    ),
                    tooltip=['Model', 'Sentiment', alt.Tooltip('Probability:Q', format='.1%')]
                ).properties(height=350)
                
                text_labels = chart_conf.mark_text(
                    align='left', baseline='middle', dx=5, color='white', fontWeight='bold', fontSize=10
                ).encode(
                    text=alt.Text('Probability:Q', format='.1%')
                )
                
                st.altair_chart(chart_conf + text_labels, use_container_width=True)

                
elif mode == "Batch Simulation":
    st.header("Batch Inference Simulation")
    st.write("Execute large-scale inference to simulate real-world data processing, measure throughput latency, and evaluate classification distribution.")
    
    st.markdown("### Step 1: Configuration")
    c_col1, c_col2 = st.columns(2)
    with c_col1:
        selected_model = st.selectbox("Select Model Architecture", list(MODELS.keys()), index=3)
    with c_col2:
        data_source = st.radio("Select Dataset", ["Use Pre-loaded Test Set", "Upload Custom CSV"])
    
    with st.expander("Advanced Hardware Settings"):
        hardware_target = st.selectbox("Compute Device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
        show_live_trace = st.checkbox("Enable Randomized Execution Trace (Shows 5 random samples)", value=True)
        if data_source == "Use Pre-loaded Test Set":
            sample_size = st.slider("Number of samples to simulate", min_value=10, max_value=200, value=50, step=10)
    
    df_test = None
    if data_source == "Use Pre-loaded Test Set":
        default_data_path = os.path.join(BASE_DIR, 'data', 'simulation', 'test_batch.csv')
        
        if os.path.exists(default_data_path):
            full_df = pd.read_csv(default_data_path)
            try:
                safe_sample_size = min(sample_size, len(full_df))
                df_test = full_df.sample(n=safe_sample_size, random_state=42).copy()
            except NameError:
                df_test = full_df.copy()
        else:
            st.warning("Simulation dataset not found. Loading fallback data.")
            df_test = pd.DataFrame({"text": ["Ang ganda ng quality!", "Scammer yung seller.", "Sakto lang."] * 2})
            
    if df_test is not None:
        possible_cols = [c for c in df_test.columns if c.lower() in ['text', 'review', 'content']]
        default_index = list(df_test.columns).index(possible_cols[0]) if possible_cols else 0
        text_column = st.selectbox("Target Text Column", df_test.columns, index=default_index)
        
        st.markdown("### Step 2: Data Preprocessing & Execution")
        enable_preprocessing = st.checkbox("Apply Thesis Preprocessing (URL Removal & Gibberish Filter)", value=True)
        
        if st.button("Execute Batch Simulation", type="primary"):
            st.session_state.batch_state = {}
            tokenizer, model, engine = load_one_model(selected_model, device=hardware_target)
            if model is None:
                st.error("Model failed to load.")
            else:
                results = []
                progress_bar = st.progress(0)
                total_samples_attempted = len(df_test)
                skipped_samples = 0
                
                try:
                    run_inference("warmup", tokenizer, model, engine, selected_model, MODELS[selected_model], hardware_target)
                except Exception:
                    pass
                
                with st.spinner(f"Simulating inference..."):
                    for step, (original_idx, row) in enumerate(df_test.iterrows()):
                        raw_text = str(row[text_column])
                        
                        if enable_preprocessing:
                            processed_text = apply_thesis_preprocessing(raw_text)
                            if not processed_text:
                                skipped_samples += 1
                                progress_bar.progress((step + 1) / total_samples_attempted)
                                continue
                            text_input = processed_text
                        else:
                            text_input = raw_text
                        
                        logits, duration = run_inference(text_input, tokenizer, model, engine, selected_model, MODELS[selected_model], hardware_target)
                        probs = softmax(logits)
                        pred_class = np.argmax(probs)
                        labels = ["Negative", "Neutral", "Positive"]
                        sentiment_result = labels[pred_class]
                        
                        results.append({
                            "Original_Index": original_idx + 1,
                            "Original_Text": raw_text,
                            "Processed_Text": text_input if enable_preprocessing else "N/A",
                            "Predicted_Class": sentiment_result,
                            "Confidence": max(probs),
                            "Latency_ms": duration
                        })
                        progress_bar.progress((step + 1) / total_samples_attempted)
                
                df_results = pd.DataFrame(results)
                st.session_state.batch_state['df'] = df_results
                st.session_state.batch_state['total_samples'] = len(results)
                st.session_state.batch_state['skipped'] = skipped_samples
                st.session_state.batch_state['selected_model'] = selected_model
                
                # --- NEW: SELECT RANDOM SAMPLES FOR TRACE ---
                if not df_results.empty and show_live_trace:
                    st.session_state.batch_state['trace_df'] = df_results.sample(n=min(5, len(df_results))).copy()

        # --- Render from Memory if Exists ---
        if st.session_state.batch_state is not None:
            df_results = st.session_state.batch_state['df']
            total_samples = st.session_state.batch_state['total_samples']
            skipped_samples = st.session_state.batch_state['skipped']
            selected_model = st.session_state.batch_state['selected_model']
            
            st.success("Simulation Complete.")
            
            # --- RENDER RANDOMIZED TRACE ---
            if 'trace_df' in st.session_state.batch_state:
                st.subheader("Randomized Execution Trace")
                for _, row in st.session_state.batch_state['trace_df'].iterrows():
                    color = "#2ecc71" if row['Predicted_Class'] == "Positive" else "#e74c3c" if row['Predicted_Class'] == "Negative" else "#95a5a6"
                    tag = f"[{row['Predicted_Class'].upper()}]"
                    st.markdown(f"""
                    <div style='background-color: #1E1E1E; padding: 12px; border-left: 5px solid {color}; border-radius: 4px; margin-bottom: 8px;'>
                        <span style='color: #aaaaaa; font-size: 12px;'>Sample #{row['Original_Index']} (Randomly Selected)</span><br>
                        <span style='color: white; font-size: 14px;'><b>Raw:</b> "{row['Original_Text']}"</span><br>
                        <span style='color: #3498db; font-size: 13px;'><b>Clean:</b> "{row['Processed_Text']}"</span><br>
                        <span style='color: {color}; font-weight: bold;'>{tag}</span> 
                        <span style='color: #aaaaaa; font-size: 12px;'>({row['Confidence']*100:.1f}% confidence)</span>
                    </div>
                    """, unsafe_allow_html=True)
            
            if skipped_samples > 0:
                st.warning(f"{skipped_samples} sample(s) were skipped due to the gibberish filter or insufficient word count.")
            
            st.markdown("### Step 3: Simulation Metrics")
            total_seconds = df_results['Latency_ms'].sum() / 1000 if total_samples > 0 else 0
            pos_count = len(df_results[df_results['Predicted_Class'] == 'Positive']) if total_samples > 0 else 0
            pos_rate = (pos_count / total_samples) * 100 if total_samples > 0 else 0
            avg_latency = df_results['Latency_ms'].mean() if total_samples > 0 else 0
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Valid Samples", total_samples)
            m2.metric("Positive Rate", f"{pos_rate:.1f}%")
            m3.metric("Avg Latency", f"{avg_latency:.2f} ms")
            m4.metric("Throughput", f"{1000/avg_latency:.1f} IPS" if avg_latency > 0 else "0 IPS")
            
            st.markdown("**Classification Distribution**")
            sentiment_counts = df_results['Predicted_Class'].value_counts().reset_index()
            sentiment_counts.columns = ['Predicted_Class', 'Count']
            sentiment_counts['Percentage'] = sentiment_counts['Count'] / total_samples
            sentiment_order = ['Negative', 'Neutral', 'Positive']
            
            # Categorical Bar Chart for Batch Sentiment Distribution
            dist_chart = alt.Chart(sentiment_counts).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
                x=alt.X('Predicted_Class:N', sort=sentiment_order, title=None, axis=alt.Axis(labelColor='white', labelFontSize=12)),
                y=alt.Y('Percentage:Q', axis=alt.Axis(format='%', title='Proportion', labelColor='white', titleColor='white')),
                color=alt.Color('Predicted_Class:N', 
                    scale=alt.Scale(domain=sentiment_order, range=['#e74c3c', '#95a5a6', '#2ecc71']), 
                    legend=None
                ),
                tooltip=['Predicted_Class', 'Count', alt.Tooltip('Percentage:Q', format='.1%')]
            ).properties(height=250)
            
            dist_text = dist_chart.mark_text(baseline='bottom', dy=-5, color='white', fontWeight='bold').encode(
                text=alt.Text('Percentage:Q', format='.1%')
            )
            
            st.altair_chart(dist_chart + dist_text, use_container_width=True)
            
            st.markdown("**Execution Logs**")
            display_df = df_results.copy()
            display_df["Confidence"] = display_df["Confidence"].apply(lambda x: f"{x*100:.1f}%")
            display_df["Latency_ms"] = display_df["Latency_ms"].apply(lambda x: f"{x:.2f} ms")
            st.dataframe(display_df, width="stretch", height=300)
            
            csv_data = df_results.to_csv(index=False).encode('utf-8')
            st.download_button("Download Simulation Logs (CSV)", data=csv_data, file_name=f"simulation_logs_{selected_model.replace(' ', '_').lower()}.csv", mime="text/csv")