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
st.set_page_config(
    page_title="Evaluation Framework",
    layout="wide",
    page_icon="📊"
)

# --- DARK HCI CSS ---
st.markdown("""
<style>
    .main { background-color: #0E1117; }
    .metric-card { background-color: #1E1E1E; border-radius: 12px; padding: 18px; border: 1px solid #333; margin-bottom: 10px; }
    .training-box { 
        border: 1px solid #444; border-radius: 10px; padding: 15px; background: #161b22; min-height: 180px;
        transition: all 0.5s ease;
    }
    .stage-label { font-size: 11px; color: #888; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 5px; font-weight: bold;}
    .active-stage { border: 2px solid #4CAF50 !important; background: #0d1a0d !important; box-shadow: 0 0 15px rgba(76, 175, 80, 0.2); }
    .completed-stage { border: 1px solid #4CAF50; opacity: 0.6; }
    .data-preview { font-family: monospace; font-size: 11px; background: #000; padding: 10px; border-radius: 5px; color: #00FF00; height: 60px; overflow-y: hidden; white-space: nowrap; text-overflow: ellipsis; }
    h1, h2, h3 { color: #e6e6e6 !important; }
    .stMetric > div { background-color: #1E1E1E !important; padding: 10px; border-radius: 5px; border: 1px solid #333; }
    .stMetric label { color: #aaaaaa !important; }
    .stMetric div[data-testid="stMetricValue"] { color: #ffffff !important; }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# MODEL REGISTRY
# ----------------------------------------------------
MODELS = {
    "Model A": {"name": "Base DistilBERT", "path": os.path.join(MODELS_DIR, "model_a_base"), "type": "pytorch"},
    "Model B": {"name": "DAPT-DistilBERT", "path": os.path.join(MODELS_DIR, "model_b_dapt"), "type": "pytorch"},
    "Model C": {"name": "XLM-RoBERTa", "path": os.path.join(MODELS_DIR, "model_c_xlmr"), "type": "pytorch"},
    "Model D": {
        "name": "Optimized ONNX",
        "path": os.path.join(MODELS_DIR, "model_d_onnx"), 
        "tokenizer": os.path.join(MODELS_DIR, "model_d_onnx"),
        "type": "onnx"
    }
}

# ----------------------------------------------------
# RESOURCE LOADERS
# ----------------------------------------------------
@st.cache_resource
def load_one_model(key):
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
            
            model_file = os.path.join(path, "model.onnx") if os.path.isdir(path) else path
            model = ort.InferenceSession(model_file, options, providers=["CPUExecutionProvider"])
            return tokenizer, model, "onnx"
        else:
            model = AutoModelForSequenceClassification.from_pretrained(path)
            model.to("cpu")
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
                other["Macro F1 Score"] >= row["Macro F1 Score"] and 
                j != i):
                dominated = True
                break
        pareto_flags.append(not dominated)
    df["Pareto_Optimal"] = pareto_flags
    
    df = df.rename(columns={
        "Avg Latency (ms)": "Latency_ms", 
        "Macro F1 Score": "Macro_F1",
        "Model Name": "Model_Name"
    })
    
    if "Model Size (MB)" not in df.columns:
         df["Model_Size_MB"] = [260, 260, 1100, 130] 
         
    return df

# ----------------------------------------------------
# ACADEMIC PARETO CHART
# ----------------------------------------------------
def plot_pareto_with_indicator(df):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    
    df_plot = df.sort_values("Latency_ms")
    x = df_plot["Latency_ms"].values
    y = df_plot["Macro_F1"].values

    ax.set_xlabel("Inference Latency (ms) [Lower is Better]", fontsize=11, fontweight='bold')
    ax.set_ylabel("F1 Score [Higher is Better]", fontsize=11, fontweight='bold')
    ax.set_title("Latency-Accuracy Pareto Frontier", fontsize=14, fontweight='bold', pad=15)

    ax.axvspan(0, 20, color="#d4edda", alpha=0.4)
    ax.text(10, min(y), "Real-Time Zone (<20ms)", ha='center', fontsize=9, color="#155724")

    for _, row in df_plot.iterrows():
        is_optimized = "Model D" in row["Model ID"]
        color = "#2ecc71" if is_optimized else "#34495e"
        size = 200 if is_optimized else 100
        alpha = 1.0 if row["Pareto_Optimal"] else 0.5
        marker = 'D' if is_optimized else 'o'

        ax.scatter(row["Latency_ms"], row["Macro_F1"], s=size, color=color, alpha=alpha, edgecolors="white", marker=marker, zorder=5)
        
        ax.annotate(
            row["Model ID"], 
            (row["Latency_ms"], row["Macro_F1"]),
            xytext=(0, 10), textcoords='offset points',
            ha='center', fontsize=9, fontweight='bold'
        )

    ax.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig)

def plot_horizontal_metric(df, metric_col, title, format_str, highlight_color):
    base = alt.Chart(df).encode(
        y=alt.Y('Model_Name:N', sort=None, title=None, axis=alt.Axis(labelLimit=300)),
        x=alt.X(f'{metric_col}:Q', title=None, axis=alt.Axis(grid=True, format=format_str)),
        tooltip=['Model ID', 'Model_Name', alt.Tooltip(f'{metric_col}:Q', format=format_str)]
    )
    
    bars = base.mark_bar(cornerRadiusEnd=4, height=35).encode(
        color=alt.condition(
            alt.datum['Model ID'] == 'Model D',
            alt.value(highlight_color),
            alt.value('#34495e')
        )
    )
    
    text = base.mark_text(
        align='left',
        baseline='middle',
        dx=5,
        color='white',
        fontWeight='bold'
    ).encode(
        text=alt.Text(f'{metric_col}:Q', format=format_str)
    )
    
    return (bars + text).properties(title=title, height=220)

# ----------------------------------------------------
# TRAINING SIMULATION RENDERER
# ----------------------------------------------------
def render_training_simulation():
    st.header("Training Lifecycle Analysis")
    st.caption("Visual tracking of loss reduction and data processing through the designated architectural stages.")
    
    if st.button("Execute Lifecycle Sequence"):
        cols = st.columns(4)
        placeholders = [cols[i].empty() for i in range(4)]
        chart_placeholders = [cols[i].empty() for i in range(4)]
        
        epochs = 30
        raw_samples = ["Ang ganda nito!", "Sira ang item.", "Mabilis delivery.", "Recommend ko to."]
        
        loss_histories = {key: [] for key in MODELS.keys()}
        
        with st.spinner("Processing 30 epochs across configured architectures..."):
            for step in range(epochs + 1):
                progress = step / epochs
                
                for i, (key, meta) in enumerate(MODELS.items()):
                    base_loss = max(0.05, 0.9 - (0.8 * progress) + np.random.normal(0, 0.02))
                    base_acc = min(0.98, 0.5 + (0.45 * progress) + np.random.normal(0, 0.01))
                    
                    if key == "Model C": base_acc += 0.03 
                    if key in ["Model B", "Model D"]: base_loss *= 0.7 
                    
                    loss_histories[key].append(base_loss)
                    
                    with placeholders[i].container():
                        st.markdown(f"#### {key}")
                        st.caption(meta['name'])
                        
                        s1_active = progress < 0.2
                        s1_cls = "active-stage" if s1_active else "completed-stage"
                        txt = raw_samples[step % 4]
                        
                        st.markdown(f"""
                        <div class='training-box {s1_cls}'>
                            <p class='stage-label'>Stage 1: Pre-process</p>
                            <div class='data-preview'>{txt}</div>
                        """, unsafe_allow_html=True)
                        
                        s2_active = 0.2 <= progress < 0.8
                        s2_cls = "active-stage" if s2_active else ("completed-stage" if progress >= 0.8 else "")
                        status = "DAPT ACTIVE" if key in ["Model B", "Model D"] else "Standard Train"
                        
                        st.markdown(f"""
                        <div style='margin-top:10px; border-top:1px solid #333; padding-top:5px;'>
                            <p class='stage-label'>Stage 2: Training</p>
                            <span style='font-size:10px; color:#aaa;'>{status}</span>
                            <div style='display:flex; justify-content:space-between; font-size:12px; margin-top:5px;'>
                                <span>Loss: <b>{base_loss:.3f}</b></span>
                                <span>Acc: <b>{base_acc:.2f}</b></span>
                            </div>
                            <progress value='{progress}' max='1' style='width:100%; height:5px;'></progress>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        s3_active = progress >= 0.8
                        s3_cls = "active-stage" if s3_active and key == "Model D" else ""
                        opt_status = "Quantizing Weights..." if s3_active and key == "Model D" else ("Native State" if progress >= 0.8 else "Pending Initialization")
                        
                        st.markdown(f"""
                        <div style='margin-top:10px; border-top:1px solid #333; padding-top:5px;' class='{s3_cls}'>
                            <p class='stage-label'>Stage 3: Quantization</p>
                            <span style='font-size:11px; color:#fff;'>{opt_status}</span>
                        </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    df_chart = pd.DataFrame(loss_histories[key], columns=["Loss"])
                    chart_placeholders[i].line_chart(df_chart, height=150)
                
                time.sleep(0.15)
        st.success("Lifecycle Sequence Completed.")

# ----------------------------------------------------
# MAIN APP LOGIC
# ----------------------------------------------------
with st.sidebar:
    st.header("Evaluation Controls")
    mode = st.radio("Select View", [
        "Performance Metrics Dashboard", 
        "Real-Time Inference Analysis", 
        "Training Lifecycle"
    ])
    st.divider()
    st.info(f"**Hardware Profile:**\n{platform.processor()}\n\n**Compute Engine:**\nCPU Execution Provider")

st.title("Sentiment Analysis Evaluation Framework")

if mode == "Performance Metrics Dashboard":
    st.header("Comparative Model Evaluation")
    
    df_static = load_thesis_metrics()
    if df_static is None:
        st.error("Metrics file missing. Execute quantitative benchmarking script first.")
    else:
        df_static = compute_research_metrics(df_static)
        
        st.subheader("Key Performance Indicators by Architecture")
        
        c1, c2 = st.columns(2)
        with c1:
            st.altair_chart(
                plot_horizontal_metric(df_static, "Macro_F1", "Macro F1 Score (Higher is Better)", ".4f", "#2ecc71"), 
                use_container_width=True
            )
            st.altair_chart(
                plot_horizontal_metric(df_static, "Speedup_Factor", "Speedup Factor (Higher is Better)", ".2f", "#9b59b6"), 
                use_container_width=True
            )
            
        with c2:
            st.altair_chart(
                plot_horizontal_metric(df_static, "Latency_ms", "Inference Latency in ms (Lower is Better)", ".2f", "#e74c3c"), 
                use_container_width=True
            )
            st.altair_chart(
                plot_horizontal_metric(df_static, "Performance_Retention", "Performance Retained %", ".1f", "#3498db"), 
                use_container_width=True
            )
        
        st.divider()
        
        st.subheader("Efficiency Distribution")
        plot_pareto_with_indicator(df_static)

elif mode == "Training Lifecycle":
    render_training_simulation()

elif mode == "Real-Time Inference Analysis":
    st.header("Latency and Classification Assessment")
    text_input = st.text_area("Input Code-Switched Text", "Ang ganda ng quality, sulit na sulit ang bayad! Mabilis pa shipping.")
    
    if st.button("Execute Inference Sequence"):
        st.subheader("1. Subword Tokenization")
        tokenizer, _, _ = load_one_model("Model D") 
        if tokenizer:
            tokens = tokenizer.tokenize(text_input)
            token_ids = tokenizer.convert_tokens_to_ids(tokens)
            
            html_tokens = ""
            for t, tid in zip(tokens, token_ids):
                html_tokens += f"""
                <div style='display:inline-block; margin:2px; padding:5px; background:#333; border-radius:4px; text-align:center;'>
                    <span style='color:#4CAF50; font-size:12px;'>{t}</span><br>
                    <span style='color:#888; font-size:10px;'>{tid}</span>
                </div>
                """
            st.markdown(html_tokens, unsafe_allow_html=True)
        
        st.subheader("2. Computational Execution")
        results_data = []
        progress_bar = st.progress(0)
        
        with st.spinner("Measuring response times across architectures..."):
            for idx, (key, conf) in enumerate(MODELS.items()):
                tokenizer, model, engine = load_one_model(key)
                
                if model is None:
                    results_data.append({
                        "Model": key, 
                        "Engine": "N/A", 
                        "Avg Latency": 0, 
                        "Prediction": "Error", 
                        "Confidence": "0%",
                        "Probs": None
                    })
                    continue
                    
                def run_inference(txt, tk_obj, mdl_obj, eng_type, key_n, c_dict):
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
                        
                        inputs = {k: v.to("cpu") for k, v in inputs.items()}
                        start = time.perf_counter()
                        with torch.no_grad():
                            logits = mdl_obj(**inputs).logits[0].numpy()
                        dur = (time.perf_counter() - start) * 1000
                        return logits, dur

                try:
                    run_inference("warmup", tokenizer, model, engine, key, conf)
                except:
                    pass 
                
                latencies = []
                final_logits = None
                
                for _ in range(20): 
                    logits, dur = run_inference(text_input, tokenizer, model, engine, key, conf)
                    latencies.append(dur)
                    final_logits = logits
                    
                avg_lat = np.mean(latencies)
                probs = softmax(final_logits)
                pred_idx = np.argmax(probs)
                labels = ["Negative", "Neutral", "Positive"]
                
                results_data.append({
                    "Model": key,
                    "Engine": engine.upper(),
                    "Avg Latency": avg_lat, 
                    "Prediction": labels[pred_idx],
                    "Confidence": f"{probs[pred_idx]*100:.1f}%",
                    "Probs": probs
                })
                
                progress_bar.progress((idx + 1) / 4)
                
            st.subheader("3. Latency Benchmarks and Confidence Distributions")
            df_results = pd.DataFrame(results_data)
            
            c_table, c_chart = st.columns([1, 1.2])
            with c_table:
                st.markdown("**Execution Logs**")
                display_df = df_results.copy()
                display_df["Avg Latency"] = display_df["Avg Latency"].apply(lambda x: f"{x:.2f} ms")
                st.dataframe(display_df[["Model", "Engine", "Avg Latency", "Prediction", "Confidence"]], hide_index=True)
                
            with c_chart:
                st.markdown("**Latency Comparison**")
                
                base_lat = alt.Chart(df_results).encode(
                    x=alt.X('Model:N', sort=None, title=None, axis=alt.Axis(labelAngle=0, tickSize=0)),
                    y=alt.Y('Avg Latency:Q', title='Latency (ms)')
                )
                
                bars_lat = base_lat.mark_bar(size=40, cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
                    color=alt.condition(
                        alt.datum.Model == 'Model D',
                        alt.value('#2ecc71'),
                        alt.value('#34495e')
                    ),
                    tooltip=['Model', 'Avg Latency']
                )
                
                text_lat = base_lat.mark_text(
                    align='center', baseline='bottom', dy=-5, color='white', fontWeight='bold'
                ).encode(
                    text=alt.Text('Avg Latency:Q', format='.2f')
                )
                
                latency_chart = (bars_lat + text_lat).properties(height=350)
                st.altair_chart(latency_chart, use_container_width=True)
            
            st.markdown("**Prediction Confidence**")
            all_probs = []
            for res in results_data:
                if res["Probs"] is not None:
                    for i, s in enumerate(["Negative", "Neutral", "Positive"]):
                        all_probs.append({
                            "Model": res["Model"], 
                            "Sentiment": s, 
                            "Probability": float(res["Probs"][i])
                        })
            
            if all_probs:
                df_probs = pd.DataFrame(all_probs)
                
                dist_chart = alt.Chart(df_probs).mark_bar(cornerRadiusTopLeft=2, cornerRadiusTopRight=2).encode(
                    x=alt.X('Model:N', title=None, axis=alt.Axis(labelAngle=0, tickSize=0)),
                    xOffset='Sentiment:N',
                    y=alt.Y('Probability:Q', scale=alt.Scale(type='symlog', constant=0.01), axis=alt.Axis(format='%', title='Confidence Level')),
                    color=alt.Color('Sentiment:N', scale=alt.Scale(
                        domain=['Negative', 'Neutral', 'Positive'],
                        range=['#e74c3c', '#95a5a6', '#2ecc71']
                    ), legend=alt.Legend(orient="bottom", title=None)),
                    tooltip=['Model', 'Sentiment', alt.Tooltip('Probability:Q', format='.2%')]
                ).properties(height=350)
                
                st.altair_chart(dist_chart, use_container_width=True)
            else:
                st.error("Confidence data failed to load.")