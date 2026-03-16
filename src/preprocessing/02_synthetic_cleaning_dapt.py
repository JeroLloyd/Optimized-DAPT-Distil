import pandas as pd
import numpy as np
import os
import sys
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- PATH CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(SCRIPT_DIR)
BASE_DIR = os.path.dirname(SRC_DIR)
RAW_DIR = os.path.join(BASE_DIR, 'data', '01_raw')
INTERIM_DIR = os.path.join(BASE_DIR, 'data', '02_interim')
REPORT_DIR = os.path.join(INTERIM_DIR, 'dedup_reports')

os.makedirs(INTERIM_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

def main():
    print(f"Project Root Detected: {BASE_DIR}")
    
    # 1. LOAD DATA
    input_path = os.path.join(RAW_DIR, "raw_synthetic_data.csv")
    if not os.path.exists(input_path):
        print(f"[ERROR] Missing {input_path}")
        return

    print(f"Loading: {input_path}")
    df = pd.read_csv(input_path)
    
    # 2. SEMANTIC DEDUPLICATION (LaBSE > 0.85)
    print("Loading LaBSE model...")
    model = SentenceTransformer('sentence-transformers/LaBSE')
    
    print("Encoding...")
    embeddings = model.encode(df['review'].tolist(), show_progress_bar=True)
    
    print("Computing Similarity Matrix...")
    sim_matrix = cosine_similarity(embeddings)
    
    threshold = 0.85
    to_drop = set()
    report_data = []

    print("Identifying duplicates...")
    for i in tqdm(range(len(sim_matrix)), desc="Deduplication"):
        if i in to_drop:
            continue
        for j in range(i + 1, len(sim_matrix)):
            score = sim_matrix[i][j]
            if score > threshold:
                to_drop.add(j)
                if score < 0.99:
                    report_data.append({
                        "Similarity Score": f"{score:.4f}",
                        "Original Sentence (Keep)": df.iloc[i]['review'],
                        "Duplicate Sentence (Remove)": df.iloc[j]['review']
                    })

    # 3. SAVE RESULTS
    if report_data:
        pd.DataFrame(report_data).to_csv(os.path.join(REPORT_DIR, "synthetic_deduplication_appendix.csv"), index=False)

    df_final = df.iloc[[i for i in range(len(df)) if i not in to_drop]].reset_index(drop=True)
    output_path = os.path.join(INTERIM_DIR, "cleaned_synthetic_data.csv")
    df_final.to_csv(output_path, index=False)
    
    print(f"SUCCESS: Saved cleaned Synthetic corpus to {output_path}")

if __name__ == "__main__":
    main()