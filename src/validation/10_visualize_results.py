import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import numpy as np
from scipy.interpolate import make_interp_spline

# --- PATH CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(SCRIPT_DIR)
BASE_DIR = os.path.dirname(SRC_DIR)

METRICS_PATH = os.path.join(BASE_DIR, 'reports', 'metrics', 'final_metrics.csv')
BENCHMARK_PATH = os.path.join(BASE_DIR, 'reports', 'metrics', 'benchmark_results_academic.csv')
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
    print("[STEP 10] Generating FINAL Thesis Visualizations & Source Data")
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
    rename_map = {
        "Macro F1 Score": "Macro_F1",
        "Avg Latency (ms)": "Latency_ms",
        "Speedup Factor": "Speedup"
    }
    df = df.rename(columns=rename_map)

    # Handle Storage_MB collision safely
    if "Model Size (MB)" in df.columns:
        if "Storage_MB" in df.columns:
            df["Storage_MB"] = df["Model Size (MB)"].fillna(df["Storage_MB"])
            df = df.drop(columns=["Model Size (MB)"])
        else:
            df = df.rename(columns={"Model Size (MB)": "Storage_MB"})
    
    # --- CALCULATE DERIVED METRICS ---
    if "Macro_F1" in df.columns:
        best_f1 = df['Macro_F1'].max()
        df['Retention_Pct'] = (df['Macro_F1'] / best_f1) * 100

    df['Short_Name'] = df['Model Name'].apply(lambda x: x.replace("Model ", "").replace("DistilBERT", "Distil"))
    
    set_professional_style()
    
    def get_colors(names, highlight_color='#27ae60'):
        return ['#7f8c8d' if 'Optimized' not in x else highlight_color for x in names]

    # --- PLOTTING & CSV GENERATION ---
    
    # # 1. Macro F1
    if "Macro_F1" in df.columns:
        dir_01 = os.path.join(FIGURES_DIR, "01_macro_f1_score")
        os.makedirs(dir_01, exist_ok=True)
        
        csv_path = os.path.join(dir_01, "1_macro_f1_score.csv")
        df[["Model Name", "Short_Name", "Macro_F1"]].to_csv(csv_path, index=False)
        print(f"Saved Data: {csv_path}")
    
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(
            x="Short_Name", y="Macro_F1", data=df, 
            hue="Short_Name", palette=get_colors(df['Model Name']), legend=False
        )
        plt.title("Model Accuracy (Macro F1 Score)", fontsize=16, fontweight='bold', pad=20)
        plt.ylabel("Macro F1")
        min_f1 = df['Macro_F1'].min()
        plt.ylim(min_f1 * 0.95, 1.0) 
        
        for p in ax.patches:
            if p.get_height() > 0:
                ax.annotate(f'{p.get_height():.4f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='bottom', fontsize=11, fontweight='bold', xytext=(0, 5), textcoords='offset points')
        plt.savefig(os.path.join(dir_01, "1_macro_f1_score.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved Image: 1_macro_f1_score.png")

    # # 2. Latency
    if "Latency_ms" in df.columns:
        dir_02 = os.path.join(FIGURES_DIR, "02_inference_latency")
        os.makedirs(dir_02, exist_ok=True)
    
        csv_path = os.path.join(dir_02, "2_inference_latency.csv")
        df[["Model Name", "Short_Name", "Latency_ms"]].to_csv(csv_path, index=False)
        print(f"Saved Data: {csv_path}")
    
        plt.figure(figsize=(10, 6))
        colors = ['#95a5a6' if 'Optimized' not in x else '#2ecc71' for x in df['Model Name']]
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
        plt.savefig(os.path.join(dir_02, "2_inference_latency.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved Image: 2_inference_latency.png")

    # # 3. Model Size
    if "Storage_MB" in df.columns:
        dir_03 = os.path.join(FIGURES_DIR, "03_model_size")
        os.makedirs(dir_03, exist_ok=True)
    
        csv_path = os.path.join(dir_03, "3_model_size.csv")
        df[["Model Name", "Short_Name", "Storage_MB"]].to_csv(csv_path, index=False)
        print(f"Saved Data: {csv_path}")
    
        plt.figure(figsize=(10, 6))
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
        plt.savefig(os.path.join(dir_03, "3_model_size.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved Image: 3_model_size.png")
    else:
        print("[SKIP] Fig 3 skipped (No storage data found).")

    # 4. Architectural Speedup Comparisons (Figures 8 & 9)
    if "Latency_ms" in df.columns:
        dir_04 = os.path.join(FIGURES_DIR, "04_speedup_factor")
        os.makedirs(dir_04, exist_ok=True)

        def get_lat(model_name_target):
            row = df[df['Model Name'].str.contains(model_name_target, case=False, na=False)]
            return row['Latency_ms'].values[0] if not row.empty else None
            
        # Target the exact strings saved by Script 08
        lat_b = get_lat("DAPT-DistilBERT")
        lat_c = get_lat("XLM-R")
        lat_d = get_lat("Optimized DAPT")

        # 4A. Figure 8: Intra-Architecture Speedup (Model B vs Model D)
        if lat_b is not None and lat_d is not None:
            speedup_intra = lat_b / lat_d
            df_intra = df[df['Model Name'].str.contains("DAPT-DistilBERT|Optimized DAPT", case=False, na=False)].copy()
            
            # Save corresponding CSV
            csv_intra = os.path.join(dir_04, "intra_architecture_speedup.csv")
            df_intra[["Model Name", "Short_Name", "Latency_ms"]].to_csv(csv_intra, index=False)
            print(f"Saved Data: {csv_intra}")
            
            plt.figure(figsize=(8, 6))
            ax = sns.barplot(x="Short_Name", y="Latency_ms", data=df_intra, palette=['#3498db', '#2ecc71'])
            plt.title("Intra-Architecture Speedup\n(32-bit vs 8-bit Quantization)", fontsize=16, fontweight='bold', pad=20)
            plt.ylabel("Inference Latency (ms)", fontweight='bold')
            plt.xlabel("")
            
            plt.annotate(f"{speedup_intra:.2f}x Faster", 
                         xy=(0.5, 0.75), xycoords='axes fraction', 
                         ha='center', fontsize=14, fontweight='bold', color='#c0392b',
                         bbox=dict(boxstyle="round,pad=0.4", fc="#fadbd8", ec="#c0392b", lw=2))
                         
            for p in ax.patches:
                 if p.get_height() > 0:
                    ax.annotate(f'{p.get_height():.2f} ms', (p.get_x() + p.get_width() / 2., p.get_height()),
                                ha='center', va='bottom', fontsize=12, fontweight='bold', xytext=(0, 5), textcoords='offset points')
            
            intra_path = os.path.join(dir_04, "intra_architecture_speedup.png")
            plt.savefig(intra_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved Image: intra_architecture_speedup.png ({speedup_intra:.2f}x)")
        else:
            print("[SKIP] Intra-Architecture Speedup (Missing latency data for DAPT-DistilBERT or Optimized DAPT).")

        # 4B. Figure 9: Inter-Architecture Benchmarking (Model C vs Model D)
        if lat_c is not None and lat_d is not None:
            speedup_inter = lat_c / lat_d
            df_inter = df[df['Model Name'].str.contains("XLM-R|Optimized DAPT", case=False, na=False)].copy()
            
            # Save corresponding CSV
            csv_inter = os.path.join(dir_04, "inter_architecture_benchmarking.csv")
            df_inter[["Model Name", "Short_Name", "Latency_ms"]].to_csv(csv_inter, index=False)
            print(f"Saved Data: {csv_inter}")
            
            plt.figure(figsize=(8, 6))
            ax = sns.barplot(x="Short_Name", y="Latency_ms", data=df_inter, palette=['#e67e22', '#2ecc71'])
            plt.title("Inter-Architecture Benchmarking\n(Large-Scale vs Edge-Optimized)", fontsize=16, fontweight='bold', pad=20)
            plt.ylabel("Inference Latency (ms)", fontweight='bold')
            plt.xlabel("")
            
            plt.annotate(f"{speedup_inter:.2f}x Faster", 
                         xy=(0.5, 0.75), xycoords='axes fraction', 
                         ha='center', fontsize=14, fontweight='bold', color='#c0392b',
                         bbox=dict(boxstyle="round,pad=0.4", fc="#fadbd8", ec="#c0392b", lw=2))
                         
            for p in ax.patches:
                 if p.get_height() > 0:
                    ax.annotate(f'{p.get_height():.2f} ms', (p.get_x() + p.get_width() / 2., p.get_height()),
                                ha='center', va='bottom', fontsize=12, fontweight='bold', xytext=(0, 5), textcoords='offset points')
            
            inter_path = os.path.join(dir_04, "inter_architecture_benchmarking.png")
            plt.savefig(inter_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved Image: inter_architecture_benchmarking.png ({speedup_inter:.2f}x)")
        else:
            print("[SKIP] Inter-Architecture Benchmarking (Missing latency data for XLM-R or Optimized DAPT).")

    print(f"\n[COMPLETE] Speedup Figures and CSV files saved to: {dir_04}")

    # # 5. Retention
    if "Retention_Pct" in df.columns:
        dir_05 = os.path.join(FIGURES_DIR, "05_performance_retention")
        os.makedirs(dir_05, exist_ok=True)
    
        csv_path = os.path.join(dir_05, "5_performance_retention.csv")
        df[["Model Name", "Short_Name", "Retention_Pct"]].to_csv(csv_path, index=False)
        print(f"Saved Data: {csv_path}")
    
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(
            x="Short_Name", y="Retention_Pct", data=df, 
            hue="Short_Name", palette=get_colors(df['Model Name'], '#2980b9'), legend=False
        )
        plt.title("Accuracy Retention", fontsize=16, fontweight='bold', pad=20)
        plt.ylabel("Retention (%)")
        plt.ylim(90, 105) 
        for p in ax.patches:
             if p.get_height() > 0:
                ax.annotate(f'{p.get_height():.1f}%', (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='bottom', fontsize=11, fontweight='bold', xytext=(0, 5), textcoords='offset points')
        plt.savefig(os.path.join(dir_05, "5_performance_retention.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved Image: 5_performance_retention.png")

    # # 6. Pareto Frontier
    if "Latency_ms" in df.columns and "Macro_F1" in df.columns:
        dir_06 = os.path.join(FIGURES_DIR, "06_pareto_frontier")
        os.makedirs(dir_06, exist_ok=True)
    
        df_pareto = df.copy().sort_values("Latency_ms")
        
        csv_path = os.path.join(dir_06, "6_pareto_frontier.csv")
        df_pareto[["Model Name", "Short_Name", "Latency_ms", "Macro_F1"]].to_csv(csv_path, index=False)
        print(f"Saved Data: {csv_path}")
    
        x = df_pareto["Latency_ms"].values
        y = df_pareto["Macro_F1"].values
    
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axvspan(0, 15, color='#f0f9f1', alpha=0.8, zorder=1)
        ax.text(8, min(y) * 0.99, "Real-Time Zone\n(<15ms)", ha='center', fontsize=11, color='#155724', fontweight='bold')
    
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
    
        plt.savefig(os.path.join(dir_06, "6_pareto_frontier.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved Image: 6_pareto_frontier.png")

    print(f"\n[COMPLETE] Speedup Figures and CSV files saved to: {dir_04}")

if __name__ == "__main__":
    generate_visualizations()