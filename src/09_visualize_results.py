import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import numpy as np

# Ensure we can find config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    import config
    RESULTS_DIR = config.RESULTS_DIR
except AttributeError:
    RESULTS_DIR = "results"

def set_professional_style():
    """Sets a high-end academic visualization style."""
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 12,
        'axes.edgecolor': '#333333',
        'axes.linewidth': 1.2,
        'grid.color': '#e0e0e0',
        'grid.linestyle': '--',
        'grid.alpha': 0.6
    })

def generate_visualizations():
    print("="*40)
    print("[STEP 9] Generating FINAL Thesis Visualizations...")
    print("="*40)

    # 1. Load Data
    results_path = os.path.join(RESULTS_DIR, "metrics_summary.csv")
    if not os.path.exists(results_path):
        print(f"[ERROR] Could not find {results_path}. Run Step 7 first.")
        return

    df = pd.read_csv(results_path)
    
    # Calculate extra metrics
    best_f1 = df['Macro_F1'].max()
    df['Retention_Pct'] = (df['Macro_F1'] / best_f1) * 100
    
    # Short names for cleaner charts
    df['Short_Name'] = df['Model_Name'].apply(lambda x: x.replace("Model ", "").replace("DistilBERT", "Distil").replace("Optimized DAPT", "Model D (Optimized)"))

    figures_dir = os.path.join(RESULTS_DIR, "figures")
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)

    set_professional_style()

    # Define Colors: Grays for baselines, Gold/Green for Model D
    def get_colors(names, highlight_color='#f1c40f'):
        return ['#7f8c8d' if 'Optimized' not in x else highlight_color for x in names]

    # ==========================================
    # 1. MACRO F1-SCORE
    # ==========================================
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x="Short_Name", y="Macro_F1", data=df, palette=get_colors(df['Model_Name'], '#27ae60'))
    plt.title("Model Accuracy (Macro F1 Score)", fontsize=16, fontweight='bold')
    plt.ylabel("Macro F1 [Higher is Better]")
    plt.ylim(0.7, 0.9)
    plt.xticks(rotation=15)
    
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.4f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
    plt.savefig(os.path.join(figures_dir, "1_macro_f1_score.png"), dpi=300, bbox_inches='tight')
    print("[SAVED] 1_macro_f1_score.png")

    # ==========================================
    # 2. INFERENCE LATENCY
    # ==========================================
    plt.figure(figsize=(10, 6))
    colors = ['#95a5a6' if 'Optimized' not in x else '#2ecc71' for x in df['Model_Name']]
    ax = sns.barplot(x="Short_Name", y="Latency_ms", data=df, palette=colors)
    
    plt.title("Inference Speed (Latency)", fontsize=16, fontweight='bold')
    plt.ylabel("Latency (ms) [Lower is Better]")
    plt.xticks(rotation=15)

    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f} ms', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
    plt.savefig(os.path.join(figures_dir, "2_inference_latency.png"), dpi=300, bbox_inches='tight')
    print("[SAVED] 2_inference_latency.png")

    # ==========================================
    # 3. MODEL SIZE
    # ==========================================
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x="Short_Name", y="Model_Size_MB", data=df, palette=colors)
    plt.title("Storage Efficiency (Model Size)", fontsize=16, fontweight='bold')
    plt.ylabel("Size (MB) [Lower is Better]")
    plt.xticks(rotation=15)

    for p in ax.patches:
        ax.annotate(f'{p.get_height():.0f} MB', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
    plt.savefig(os.path.join(figures_dir, "3_model_size.png"), dpi=300, bbox_inches='tight')
    print("[SAVED] 3_model_size.png")

    # ==========================================
    # 4. SPEEDUP FACTOR
    # ==========================================
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x="Short_Name", y="Speedup_Factor", data=df, palette=get_colors(df['Model_Name'], '#8e44ad'))
    plt.title("Speedup Factor vs. Baseline", fontsize=16, fontweight='bold')
    plt.ylabel("Speedup Multiplier (x)")
    plt.axhline(1.0, color='red', linestyle='--', label="Baseline Reference")
    plt.xticks(rotation=15)
    
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{height:.2f}x', (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
    plt.savefig(os.path.join(figures_dir, "4_speedup_factor.png"), dpi=300, bbox_inches='tight')
    print("[SAVED] 4_speedup_factor.png")

    # ==========================================
    # 5. PERFORMANCE RETENTION
    # ==========================================
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x="Short_Name", y="Retention_Pct", data=df, palette=get_colors(df['Model_Name'], '#2980b9'))
    plt.title("Performance Retention (Relative to Best Model)", fontsize=16, fontweight='bold')
    plt.ylabel("Retention (%)")
    plt.ylim(90, 100.5)
    plt.xticks(rotation=15)

    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1f}%', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
    plt.savefig(os.path.join(figures_dir, "5_performance_retention.png"), dpi=300, bbox_inches='tight')
    print("[SAVED] 5_performance_retention.png")

    # ==========================================
    # 6. PARETO FRONTIER (DUAL-AXIS VERSION)
    # ==========================================
    # The "Easier Visualization" using Bars for Speed and Line for Accuracy
    
    # 1. Sort A -> B -> C -> D
    desired_order = ["Model A Base DistilBERT", "Model B DAPT-DistilBERT", "Model C XLM-RoBERTa", "Model D Optimized DAPT"]
    df['Sort_Key'] = df['Model_Name'].apply(lambda x: desired_order.index(x) if x in desired_order else 99)
    df_sorted = df.sort_values('Sort_Key')
    short_names_sorted = ["Model A\n(Base)", "Model B\n(DAPT)", "Model C\n(XLM-R)", "Model D\n(Proposed)"]

    fig, ax1 = plt.subplots(figsize=(11, 7))
    
    # --- PLOT BARS (LATENCY) ---
    colors = ['#bdc3c7', '#bdc3c7', '#95a5a6', '#2ecc71'] 
    bars = ax1.bar(short_names_sorted, df_sorted['Latency_ms'], color=colors, alpha=0.9, width=0.5)
    
    ax1.set_ylabel('Inference Speed (ms) [Lower is Better]', fontsize=13, fontweight='bold', color='#34495e')
    ax1.tick_params(axis='y', labelcolor='#34495e', labelsize=11)
    ax1.set_ylim(0, df_sorted['Latency_ms'].max() * 1.3) # Headroom for line chart

    # --- FIX: LABELS INSIDE BARS (Centered) ---
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height/2, f'{height:.1f} ms', 
                 ha='center', va='center', fontsize=12, fontweight='bold', color='white',
                 bbox=dict(facecolor='black', alpha=0.1, edgecolor='none', pad=0))

    # --- PLOT LINE (ACCURACY) ---
    ax2 = ax1.twinx()
    ax2.plot(short_names_sorted, df_sorted['Macro_F1'], color='#e74c3c', marker='o', markersize=12, linewidth=4)
    
    ax2.set_ylabel('Accuracy (F1) [Higher is Better]', fontsize=13, fontweight='bold', color='#c0392b')
    ax2.tick_params(axis='y', labelcolor='#c0392b', labelsize=11)
    ax2.set_ylim(0.70, 0.95) # Zoom in

    # --- LINE LABELS (Above points) ---
    for i, txt in enumerate(df_sorted['Macro_F1']):
        ax2.annotate(f"{txt:.2f}", (i, txt), xytext=(0, 20), textcoords='offset points', 
                     ha='center', fontsize=12, fontweight='bold', color='#c0392b',
                     bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#e74c3c", lw=2))

    plt.title("Pareto Analysis: Efficiency vs. Effectiveness", fontsize=16, fontweight='bold', pad=20)
    
    # Legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor='#bdc3c7', label='Latency (Baselines)'),
        Patch(facecolor='#2ecc71', label='Latency (Model D - Fastest)'),
        Line2D([0], [0], color='#e74c3c', lw=4, marker='o', label='Accuracy Trend')
    ]
    plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False, fontsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "6_pareto_frontier.png"), dpi=300)
    print("[SAVED] 6_pareto_frontier.png (Dual-Axis Version)")

    print(f"\n[SUCCESS] All 6 Figures generated in '{figures_dir}'")

if __name__ == "__main__":
    generate_visualizations()