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
    """Sets a high-end academic visualization style for Thesis."""
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
    
    # Calculate extra metrics for Chapter 4
    best_f1 = df['Macro_F1'].max()
    df['Retention_Pct'] = (df['Macro_F1'] / best_f1) * 100
    
    # Short names for cleaner charts
    df['Short_Name'] = df['Model_Name'].apply(lambda x: x.replace("Model ", "").replace("DistilBERT", "Distil").replace("Optimized DAPT", "Model D (Optimized)"))

    figures_dir = os.path.join(RESULTS_DIR, "figures")
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)

    set_professional_style()

    # Define Colors: Grays for baselines, Gold/Green for Model D (The Highlight)
    def get_colors(names, highlight_color='#f1c40f'):
        return ['#7f8c8d' if 'Optimized' not in x else highlight_color for x in names]

    # ==========================================
    # 1. MACRO F1-SCORE
    # ==========================================
    plt.figure(figsize=(10, 6))
    # Green highlight for Model D to show it's competitive
    ax = sns.barplot(x="Short_Name", y="Macro_F1", data=df, palette=get_colors(df['Model_Name'], '#27ae60'))
    plt.title("Model Accuracy (Macro F1 Score)", fontsize=16, fontweight='bold')
    plt.ylabel("Macro F1 [Higher is Better]")
    plt.ylim(0.7, 0.9) # Zoom in to show differences
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
    # Green highlight for Model D to show speed
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
    # Purple highlight for the massive speedup
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
    plt.ylim(90, 100.5) # Focus on the top 10%
    plt.xticks(rotation=15)

    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1f}%', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
    plt.savefig(os.path.join(figures_dir, "5_performance_retention.png"), dpi=300, bbox_inches='tight')
    print("[SAVED] 5_performance_retention.png")

    # ==========================================
    # 6. PARETO FRONTIER (MATCHING SECOND IMAGE)
    # ==========================================
    from scipy.interpolate import make_interp_spline

    # 1. Sort data by Latency to prevent line-crossing
    df_pareto = df.copy().sort_values("Latency_ms")
    x = df_pareto["Latency_ms"].values
    y = df_pareto["Macro_F1"].values

    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 2. Real-Time Deployment Zone Overlay
    ax.axvspan(0, 20, color='#f0f9f1', alpha=0.8, zorder=1)
    ax.text(10, 0.785, "Real-Time Zone\n(<20ms)", 
            ha='center', fontsize=11, color='#155724', fontweight='bold')

    # 3. Generate Linear Frontier Line (k=1 fixes the "dip")
    if len(x) > 1:
        x_smooth = np.linspace(x.min(), x.max(), 300)
        spline = make_interp_spline(x, y, k=1) # Linear connection
        ax.plot(x_smooth, spline(x_smooth), color='#003366', linewidth=4, zorder=2)

    # 4. Plot Individual Model Points
    for _, row in df_pareto.iterrows():
        is_proposed = "Optimized" in row["Model_Name"]
        color = '#e74c3c' if is_proposed else '#34495e'
        marker = 'D' if is_proposed else 'o'
        
        ax.scatter(row["Latency_ms"], row["Macro_F1"], color=color, marker=marker, 
                   s=250 if is_proposed else 180, zorder=5, edgecolors='white', linewidth=1.5)
        
        # Labels positioned above points
        ax.annotate(row["Short_Name"], 
                    xy=(row["Latency_ms"], row["Macro_F1"]),
                    xytext=(0, 12), textcoords='offset points',
                    ha='center', fontsize=11, fontweight='bold' if is_proposed else 'normal')

    # 5. Professional Styling
    ax.set_title("Latency–Accuracy Pareto Frontier", fontsize=18, fontweight='bold', pad=25)
    ax.set_xlabel("Inference Latency (ms) [Lower is Better]", fontsize=13, fontweight='bold', labelpad=12)
    ax.set_ylabel("Macro F1 Score [Higher is Better]", fontsize=13, fontweight='bold', labelpad=12)
    
    # Precision Axes Limits
    ax.set_xlim(0, 160)
    ax.set_ylim(0.76, 0.90) 
    ax.grid(True, linestyle='--', color='#ecf0f1', alpha=0.7)

    plt.savefig(os.path.join(figures_dir, "6_pareto_frontier.png"), dpi=300, bbox_inches='tight')
    print("[SUCCESS] Pareto Frontier updated to match target style (Linear).")
    print(f"\n[SUCCESS] All 6 Figures generated in '{figures_dir}'")

if __name__ == "__main__":
    generate_visualizations()