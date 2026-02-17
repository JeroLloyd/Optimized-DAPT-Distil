import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import numpy as np
from scipy.interpolate import make_interp_spline

# --- PATH CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)

METRICS_PATH = os.path.join(BASE_DIR, 'reports', 'metrics', 'final_metrics.csv')
BENCHMARK_PATH = os.path.join(BASE_DIR, 'reports', 'metrics', 'benchmark_results.csv')
FIGURES_DIR = os.path.join(BASE_DIR, 'reports', 'figures')

os.makedirs(FIGURES_DIR, exist_ok=True)

def set_professional_style():
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
    print("="*60)
    print("[STEP 10] Generating FINAL Thesis Visualizations (REAL DATA ONLY)")
    print("="*60)

    # 1. Load Data
    if not os.path.exists(METRICS_PATH):
        print(f"[ERROR] {METRICS_PATH} missing. Run Script 08.")
        return
    df = pd.read_csv(METRICS_PATH)
    
    # Load Benchmark Data if available (Optional merge)
    if os.path.exists(BENCHMARK_PATH):
        print(f"Loading benchmark data from {BENCHMARK_PATH}...")
        df_bench = pd.read_csv(BENCHMARK_PATH)
        # Merge on 'Model Name'
        df = pd.merge(df, df_bench, on="Model Name", how="left")
    else:
        print(f"[WARNING] {BENCHMARK_PATH} missing. Relying on Script 08 metrics.")

    # --- FIX: ROBUST COLUMN MAPPING & DEDUPLICATION ---
    # We rename columns one by one to avoid collisions
    rename_map = {
        "Macro F1 Score": "Macro_F1",
        "Avg Latency (ms)": "Latency_ms",
        "Speedup Factor": "Speedup"
    }
    df = df.rename(columns=rename_map)

    # Handle Storage_MB collision safely
    # If "Storage_MB" exists (from benchmark) AND "Model Size (MB)" exists (from metrics)
    if "Model Size (MB)" in df.columns:
        if "Storage_MB" in df.columns:
            # We have both. Prefer 'Model Size (MB)' from metrics as primary, fill with Benchmark.
            df["Storage_MB"] = df["Model Size (MB)"].fillna(df["Storage_MB"])
            df = df.drop(columns=["Model Size (MB)"])
        else:
            # Only metrics has it, just rename
            df = df.rename(columns={"Model Size (MB)": "Storage_MB"})
    
    # --- CALCULATE DERIVED METRICS ---
    if "Macro_F1" in df.columns:
        best_f1 = df['Macro_F1'].max()
        df['Retention_Pct'] = (df['Macro_F1'] / best_f1) * 100
    
    # Recalculate Speedup if missing (Relative to slowest model)
    if "Latency_ms" in df.columns and "Speedup" not in df.columns:
        baseline_lat = df['Latency_ms'].max()
        df['Speedup'] = baseline_lat / df['Latency_ms']

    # Short names for cleaner plots
    df['Short_Name'] = df['Model Name'].apply(lambda x: x.replace("Model ", "").replace("DistilBERT", "Distil"))
    
    set_professional_style()
    
    def get_colors(names, highlight_color='#27ae60'):
        return ['#7f8c8d' if 'Optimized' not in x else highlight_color for x in names]

    # --- PLOTTING FUNCTIONS ---
    
    # 1. Macro F1
    if "Macro_F1" in df.columns:
        plt.figure(figsize=(10, 6))
        # Fixed: Added hue and legend=False to silence warning
        ax = sns.barplot(
            x="Short_Name", y="Macro_F1", data=df, 
            hue="Short_Name", palette=get_colors(df['Model Name']), legend=False
        )
        plt.title("Model Accuracy (Macro F1 Score)", fontsize=16, fontweight='bold', pad=20)
        plt.ylabel("Macro F1")
        
        # Dynamic Y-lim to show differences clearly
        min_f1 = df['Macro_F1'].min()
        plt.ylim(min_f1 * 0.95, 1.0) 
        
        for p in ax.patches:
            if p.get_height() > 0:
                ax.annotate(f'{p.get_height():.4f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='bottom', fontsize=11, fontweight='bold', xytext=(0, 5), textcoords='offset points')
        plt.savefig(os.path.join(FIGURES_DIR, "1_macro_f1_score.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved Fig 1.")

    # 2. Latency
    if "Latency_ms" in df.columns:
        plt.figure(figsize=(10, 6))
        colors = ['#95a5a6' if 'Optimized' not in x else '#2ecc71' for x in df['Model Name']]
        # Fixed: Added hue and legend=False
        ax = sns.barplot(
            x="Short_Name", y="Latency_ms", data=df, 
            hue="Short_Name", palette=colors, legend=False
        )
        plt.title("Inference Speed (Latency)", fontsize=16, fontweight='bold', pad=20)
        plt.ylabel("Latency (ms)")
        for p in ax.patches:
             if p.get_height() > 0:
                ax.annotate(f'{p.get_height():.2f} ms', (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='bottom', fontsize=11, fontweight='bold', xytext=(0, 5), textcoords='offset points')
        plt.savefig(os.path.join(FIGURES_DIR, "2_inference_latency.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved Fig 2.")

    # 3. Model Size
    if "Storage_MB" in df.columns:
        plt.figure(figsize=(10, 6))
        # Fixed: Added hue and legend=False
        ax = sns.barplot(
            x="Short_Name", y="Storage_MB", data=df, 
            hue="Short_Name", palette=get_colors(df['Model Name']), legend=False
        )
        plt.title("Storage Efficiency (Disk Usage)", fontsize=16, fontweight='bold', pad=20)
        plt.ylabel("Size (MB)")
        for p in ax.patches:
             if p.get_height() > 0:
                ax.annotate(f'{p.get_height():.1f} MB', (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='bottom', fontsize=11, fontweight='bold', xytext=(0, 5), textcoords='offset points')
        plt.savefig(os.path.join(FIGURES_DIR, "3_model_size.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved Fig 3.")
    else:
        print("[SKIP] Fig 3 skipped (No storage data found).")

    # 4. Speedup
    if "Speedup" in df.columns:
        plt.figure(figsize=(10, 6))
        # Fixed: Added hue and legend=False
        ax = sns.barplot(
            x="Short_Name", y="Speedup", data=df, 
            hue="Short_Name", palette=get_colors(df['Model Name'], '#8e44ad'), legend=False
        )
        plt.title("Inference Speedup Factor", fontsize=16, fontweight='bold', pad=20)
        plt.ylabel("Multiplier (x)")
        plt.axhline(1.0, color='red', linestyle='--', alpha=0.5)
        for p in ax.patches:
             if p.get_height() > 0:
                ax.annotate(f'{p.get_height():.2f}x', (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='bottom', fontsize=11, fontweight='bold', xytext=(0, 5), textcoords='offset points')
        plt.savefig(os.path.join(FIGURES_DIR, "4_speedup_factor.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved Fig 4.")

    # 5. Retention
    if "Retention_Pct" in df.columns:
        plt.figure(figsize=(10, 6))
        # Fixed: Added hue and legend=False
        ax = sns.barplot(
            x="Short_Name", y="Retention_Pct", data=df, 
            hue="Short_Name", palette=get_colors(df['Model Name'], '#2980b9'), legend=False
        )
        plt.title("Accuracy Retention", fontsize=16, fontweight='bold', pad=20)
        plt.ylabel("Retention (%)")
        plt.ylim(90, 105) # Adjusted for high retention
        for p in ax.patches:
             if p.get_height() > 0:
                ax.annotate(f'{p.get_height():.1f}%', (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='bottom', fontsize=11, fontweight='bold', xytext=(0, 5), textcoords='offset points')
        plt.savefig(os.path.join(FIGURES_DIR, "5_performance_retention.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved Fig 5.")

    # 6. Pareto Frontier
    if "Latency_ms" in df.columns and "Macro_F1" in df.columns:
        df_pareto = df.copy().sort_values("Latency_ms")
        x = df_pareto["Latency_ms"].values
        y = df_pareto["Macro_F1"].values

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axvspan(0, 15, color='#f0f9f1', alpha=0.8, zorder=1)
        ax.text(8, min(y) * 0.99, "Real-Time Zone\n(<15ms)", ha='center', fontsize=11, color='#155724', fontweight='bold')

        # Smooth curve if enough points, else straight lines
        if len(x) > 2:
            try:
                spline = make_interp_spline(x, y, k=2) 
                x_smooth = np.linspace(x.min(), x.max(), 300)
                ax.plot(x_smooth, spline(x_smooth), color='#003366', linewidth=3, zorder=2, alpha=0.4)
            except:
                ax.plot(x, y, color='#003366', linewidth=3, zorder=2, alpha=0.4)
        elif len(x) > 1:
             ax.plot(x, y, color='#003366', linewidth=3, zorder=2, alpha=0.4)

        for _, row in df_pareto.iterrows():
            is_opt = "Optimized" in row["Model Name"]
            ax.scatter(row["Latency_ms"], row["Macro_F1"], color='#e74c3c' if is_opt else '#34495e', 
                       marker='D' if is_opt else 'o', s=200, zorder=5, edgecolors='white')
            
            offset = (0, 12)
            ax.annotate(row["Short_Name"], (row["Latency_ms"], row["Macro_F1"]), xytext=offset, 
                        textcoords='offset points', ha='center', fontsize=10, fontweight='bold' if is_opt else 'normal')

        ax.set_title("Latency–Accuracy Pareto Frontier", fontsize=18, fontweight='bold', pad=25)
        ax.set_xlabel("Inference Latency (ms)", fontsize=13, fontweight='bold')
        ax.set_ylabel("Macro F1 Score", fontsize=13, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.6)

        plt.savefig(os.path.join(FIGURES_DIR, "6_pareto_frontier.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved Fig 6.")

    print(f"\n[COMPLETE] All valid figures saved to: {FIGURES_DIR}")

if __name__ == "__main__":
    generate_visualizations()