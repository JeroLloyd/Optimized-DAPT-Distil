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
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

# --- PATH CONFIGURATION ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(CURRENT_DIR, 'models')):
    BASE_DIR = CURRENT_DIR
else:
    BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))

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
    "Model A": {"name": "Base DistilBERT", "path": os.path.join(MODELS_DIR, "model_a_base"), "type": "pytorch"},
    "Model B": {"name": "DAPT-DistilBERT", "path": os.path.join(MODELS_DIR, "model_b_dapt"), "type": "pytorch"},
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

@st.cache_data
def load_thesis_metrics():
    if os.path.exists(RESULTS_PATH):
        df = pd.read_csv(RESULTS_PATH)
        df.columns = [c.strip() for c in df.columns] 
        return df
    return None

def compute_research_metrics(df):
    baseline = df.loc[df["Avg Latency (ms)"].idxmax()]
    df["Speedup_Factor"] = baseline["Avg Latency (ms)"] / df["Avg Latency (ms)"]
    best_f1 = df["Macro F1 Score"].max()
    df["Performance_Retention"] = (df["Macro F1 Score"] / best_f1) * 100
    
    pareto_flags = []
    for i, row in df.iterrows():
        dominated = False
        for j, other in df.iterrows():
            if (other["Avg Latency (ms)"] <= row["Avg Latency (ms)"] and 
                other["Macro F1 Score"] >= row["Macro F1 Score"] and j != i):
                dominated = True
                break
        pareto_flags.append(not dominated)
    df["Pareto_Optimal"] = pareto_flags
    df = df.rename(columns={"Avg Latency (ms)": "Latency_ms", "Macro F1 Score": "Macro_F1", "Model Name": "Model_Name"})
    return df

# --- CHART FUNCTIONS ---
def plot_pareto_with_indicator(df):
    plt.style.use("seaborn-v0_8-darkgrid")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor('#1E1E1E')
    
    df_plot = df.sort_values("Latency_ms")
    x = df_plot["Latency_ms"].values
    y = df_plot["Macro_F1"].values

    ax.set_xlabel("Inference Latency (ms) [Lower is Better]", fontsize=11, fontweight='bold', color='white')
    ax.set_ylabel("F1 Score [Higher is Better]", fontsize=11, fontweight='bold', color='white')
    ax.set_title("Latency-Accuracy Pareto Frontier", fontsize=14, fontweight='bold', color='white', pad=15)
    ax.tick_params(colors='white')

    ax.axvspan(0, 20, color="#2ecc71", alpha=0.15)
    ax.text(10, min(y), "Real-Time Zone (<20ms)", ha='center', fontsize=9, color="#2ecc71")

    for _, row in df_plot.iterrows():
        is_optimized = "Model D" in row["Model ID"]
        color = "#2ecc71" if is_optimized else "#3498db"
        size = 200 if is_optimized else 100
        alpha = 1.0 if row["Pareto_Optimal"] else 0.5
        marker = 'D' if is_optimized else 'o'

        ax.scatter(row["Latency_ms"], row["Macro_F1"], s=size, color=color, alpha=alpha, edgecolors="white", marker=marker, zorder=5)
        ax.annotate(row["Model ID"], (row["Latency_ms"], row["Macro_F1"]), xytext=(0, 10), textcoords='offset points', ha='center', fontsize=9, fontweight='bold', color='white')

    ax.grid(True, linestyle='--', alpha=0.3)
    st.pyplot(fig)

def plot_horizontal_metric(df, metric_col, title, format_str, highlight_color):
    base = alt.Chart(df).encode(
        y=alt.Y('Model_Name:N', sort=None, title=None, axis=alt.Axis(labelLimit=300, labelColor='white')),
        x=alt.X(f'{metric_col}:Q', title=None, axis=alt.Axis(grid=True, format=format_str, labelColor='white')),
        tooltip=['Model ID', 'Model_Name', alt.Tooltip(f'{metric_col}:Q', format=format_str)]
    )
    bars = base.mark_bar(cornerRadiusEnd=4, height=35).encode(
        color=alt.condition(alt.datum['Model ID'] == 'Model D', alt.value(highlight_color), alt.value('#34495e'))
    )
    text = base.mark_text(align='left', baseline='middle', dx=5, color='white', fontWeight='bold').encode(
        text=alt.Text(f'{metric_col}:Q', format=format_str)
    )
    return (bars + text).properties(title=alt.TitleParams(text=title, color='white'), height=220)

# --- INFERENCE ENGINE ---
def run_inference(txt, tk_obj, mdl_obj, eng_type, key_n, c_dict, dev):
    if eng_type == "onnx":
        inputs = tk_obj(txt, return_tensors="np", padding=True, truncation=True, max_length=128)
        if "token_type_ids" in inputs and ("distilbert" in c_dict["name"].lower() or key_n == "Model D"):
            del inputs["token_type_ids"]
        inputs = {k: v.astype(np.int64) for k, v in inputs.items()}
        start = time.perf_counter()
        logits = mdl_obj.run(None, inputs)[0][0]
        dur = (time.perf_counter() - start) * 1000
        return logits, dur
    else:
        inputs = tk_obj(txt, return_tensors="pt", padding=True, truncation=True, max_length=128)
        if "token_type_ids" in inputs and ("distilbert" in c_dict["name"].lower() or "model_a" in key_n.lower() or "model_b" in key_n.lower()):
            del inputs["token_type_ids"]
        inputs = {k: v.to(dev) for k, v in inputs.items()}
        start = time.perf_counter()
        with torch.no_grad():
            logits = mdl_obj(**inputs).logits[0].cpu().numpy()
        dur = (time.perf_counter() - start) * 1000
        return logits, dur

# --- UI LOGIC ---
with st.sidebar:
    st.header("Evaluation Controls")
    mode = st.radio("Select View", ["Evaluation Dashboard", "Diagnostic Inference", "Batch Simulation"])
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
        plot_pareto_with_indicator(df_plot)

elif mode == "Diagnostic Inference":
    st.header("Latency and Classification Assessment (CPU vs GPU)")
    text_input = st.text_area("Input Code-Switched Text", "Ang ganda ng quality, sulit na sulit ang bayad! Mabilis pa shipping.")
    
    if st.button("Execute Inference Sequence"):
        st.subheader("1. Subword Tokenization")
        tokenizer, _, _ = load_one_model("Model D", device="cpu") 
        if tokenizer:
            tokens = tokenizer.tokenize(text_input)
            token_ids = tokenizer.convert_tokens_to_ids(tokens)
            html_tokens = ""
            for t, tid in zip(tokens, token_ids):
                html_tokens += f"<div style='display:inline-block; margin:2px; padding:5px; background:#1E1E1E; border: 1px solid #333; border-radius:4px; text-align:center;'><span style='color:#2ecc71; font-size:13px; font-weight:bold;'>{t}</span><br><span style='color:#aaaaaa; font-size:11px;'>{tid}</span></div>"
            st.markdown(html_tokens, unsafe_allow_html=True)
        
        st.subheader("2. Computational Execution (Hardware Benchmark)")
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
                    
                    results_data.append({
                        "Model": key,
                        "Hardware": device.upper(),
                        "Engine": engine.upper(),
                        "Avg Latency": avg_lat, 
                        "Prediction": labels[pred_idx],
                        "Confidence": probs[pred_idx],
                        "Probs": probs
                    })
                    current_op += 1
                    progress_bar.progress(current_op / total_operations)
                    
            st.subheader("3. Latency Benchmarks and Confidence Distributions")
            df_results = pd.DataFrame(results_data)
            
            # Display Full-Width Table
            st.markdown("**Execution Logs**")
            display_df = df_results.copy()
            display_df["Avg Latency"] = display_df["Avg Latency"].apply(lambda x: f"{x:.2f} ms")
            display_df["Confidence"] = display_df["Confidence"].apply(lambda x: f"{x*100:.1f}%")
            st.dataframe(display_df[["Model", "Hardware", "Avg Latency", "Prediction", "Confidence"]], use_container_width=True, hide_index=True)
            
            # Display Side-by-Side Charts
            c1, c2 = st.columns(2)
            
            with c1:
                st.markdown("**Hardware Latency Comparison**")
                bars_lat = alt.Chart(df_results).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
                    x=alt.X('Model:N', title=None, axis=alt.Axis(labelAngle=0, labelColor='white')),
                    xOffset='Hardware:N',
                    y=alt.Y('Avg Latency:Q', title='Latency (ms)', axis=alt.Axis(labelColor='white', titleColor='white')),
                    color=alt.Color('Hardware:N', scale=alt.Scale(domain=['CPU', 'CUDA'], range=['#3498db', '#e74c3c'])),
                    tooltip=['Model', 'Hardware', alt.Tooltip('Avg Latency:Q', format='.2f')]
                )
                text_lat = bars_lat.mark_text(
                    align='center',
                    baseline='bottom',
                    dy=-5,
                    color='white',
                    fontWeight='bold',
                    fontSize=11
                ).encode(
                    text=alt.Text('Avg Latency:Q', format='.1f')
                )
                chart_lat = (bars_lat + text_lat).properties(height=380)
                st.altair_chart(chart_lat, width="stretch")
            
            with c2:
                st.markdown("**Prediction Confidence (By Architecture)**")
                all_probs = []
                cpu_results = [res for res in results_data if res["Hardware"] == "CPU"]
                for res in cpu_results:
                    if res["Probs"] is not None:
                        for i, s in enumerate(["Negative", "Neutral", "Positive"]):
                            all_probs.append({"Model": res["Model"], "Sentiment": s, "Probability": float(res["Probs"][i])})
                
                if all_probs:
                    df_probs = pd.DataFrame(all_probs)
                    dist_chart = alt.Chart(df_probs).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
                        x=alt.X('Model:N', title=None, axis=alt.Axis(labelAngle=0, labelColor='white')),
                        xOffset='Sentiment:N',
                        y=alt.Y('Probability:Q', scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format='%', title='Confidence Level', labelColor='white', titleColor='white')),
                        color=alt.Color('Sentiment:N', scale=alt.Scale(domain=['Negative', 'Neutral', 'Positive'], range=['#e74c3c', '#95a5a6', '#2ecc71'])),
                        tooltip=['Model', 'Sentiment', alt.Tooltip('Probability:Q', format='.1%')]
                    )
                    text_conf = dist_chart.mark_text(
                        align='center',
                        baseline='bottom',
                        dy=-5,
                        color='white',
                        fontWeight='bold',
                        fontSize=10
                    ).encode(
                        text=alt.Text('Probability:Q', format='.1%')
                    )
                    chart_conf = (dist_chart + text_conf).properties(height=380)
                    st.altair_chart(chart_conf, width="stretch")

elif mode == "Batch Simulation":
    st.header("Batch Processing and Data Export")
    st.write("Execute large-scale inference to measure average latency and export performance metrics.")
    
    selected_model = st.selectbox("Select Model Architecture", list(MODELS.keys()))
    hardware_target = st.selectbox("Select Hardware Target", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
    
    data_source = st.radio("Dataset Selection", ["Use Pre-loaded Thesis Dataset", "Upload Custom CSV"])
    show_live_trace = st.checkbox("Enable Live Execution Trace (Visualizes the first 5 samples)", value=True)
    
    df_test = None
    
    if data_source == "Use Pre-loaded Thesis Dataset":
        default_data_path = os.path.join(BASE_DIR, 'data', '03_processed', 'FiReCS_Final', 'test.csv')
        if os.path.exists(default_data_path):
            full_df = pd.read_csv(default_data_path)
            sample_size = st.slider("Select Sample Size for Demonstration", min_value=10, max_value=len(full_df), value=50, step=10)
            df_test = full_df.sample(n=sample_size, random_state=42).copy()
            st.success(f"Loaded {sample_size} random samples from the FiReCS test set.")
        else:
            st.warning("Default dataset not found. Loading fallback synthetic data.")
            df_test = pd.DataFrame({
                "text": [
                    "Ang ganda ng quality, sulit na sulit ang bayad!",
                    "Mabilis ang shipping but the packaging is bad.",
                    "Wag kayo bibili dito scammer yung seller.",
                    "Okay naman siya, not too bad for the price.",
                    "Super love it! I will definitely order again."
                ]
            })
    else:
        uploaded_file = st.file_uploader("Upload CSV Dataset", type="csv")
        if uploaded_file is not None:
            df_test = pd.read_csv(uploaded_file)
            
    if df_test is not None:
        possible_cols = [c for c in df_test.columns if c.lower() in ['text', 'review', 'content']]
        default_index = list(df_test.columns).index(possible_cols[0]) if possible_cols else 0
        text_column = st.selectbox("Select Text Column", df_test.columns, index=default_index)
        
        if st.button("Execute Batch Simulation"):
            tokenizer, model, engine = load_one_model(selected_model, device=hardware_target)
            if model is None:
                st.error("Model failed to load.")
            else:
                results = []
                progress_bar = st.progress(0)
                total_samples = len(df_test)
                
                try:
                    run_inference("warmup", tokenizer, model, engine, selected_model, MODELS[selected_model], hardware_target)
                except Exception:
                    pass
                
                trace_container = st.container() if show_live_trace else None
                
                with st.spinner(f"Processing {total_samples} texts using {selected_model}..."):
                    for step, (original_idx, row) in enumerate(df_test.iterrows()):
                        text_input = str(row[text_column])
                        
                        logits, duration = run_inference(text_input, tokenizer, model, engine, selected_model, MODELS[selected_model], hardware_target)
                        probs = softmax(logits)
                        pred_class = np.argmax(probs)
                        labels = ["Negative", "Neutral", "Positive"]
                        
                        results.append({
                            "Text": text_input,
                            "Predicted_Class": labels[pred_class],
                            "Confidence": max(probs),
                            "Latency_ms": duration
                        })
                        
                        if show_live_trace and step < 5:
                            with trace_container:
                                st.markdown(f"**Sample {step + 1} Pipeline:**")
                                st.info(f"**1. Input:** {text_input}")
                                tokens = tokenizer.tokenize(text_input)
                                st.code(f"2. Tokenization:\n{tokens}", language="text")
                                st.success(f"**3. Output:** {labels[pred_class]} ({max(probs)*100:.1f}%) | **Latency:** {duration:.2f} ms")
                                st.divider()
                        
                        progress_bar.progress((step + 1) / total_samples)
                
                df_results = pd.DataFrame(results)
                st.success("Batch simulation completed.")
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Samples", total_samples)
                c2.metric("Average Latency", f"{df_results['Latency_ms'].mean():.2f} ms")
                c3.metric("P95 Latency", f"{np.percentile(df_results['Latency_ms'], 95):.2f} ms")
                
                st.dataframe(df_results, width="stretch", height=300)
                
                csv_data = df_results.to_csv(index=False).encode('utf-8')
                st.download_button("Download Simulation Data (CSV)", data=csv_data, file_name=f"simulation_{selected_model.replace(' ', '_').lower()}.csv", mime="text/csv")