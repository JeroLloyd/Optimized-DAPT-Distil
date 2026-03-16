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
INTERIM_DIR = os.path.join(BASE_DIR, 'data', '02_interim')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', '03_processed')
REPORT_DIR = os.path.join(INTERIM_DIR, 'dedup_reports')

os.makedirs(PROCESSED_DIR, exist_ok=True)

def main():
    print(f"Project Root Detected: {BASE_DIR}")

    # 1. LOAD DATASETS
    lazada_path = os.path.join(INTERIM_DIR, "cleaned_lazada_data.csv")
    synthetic_path = os.path.join(INTERIM_DIR, "cleaned_synthetic_data.csv")
    
    if not (os.path.exists(lazada_path) and os.path.exists(synthetic_path)):
        print("[ERROR] Intermediate files missing. Run scripts 01 and 02.")
        return

    df_lazada = pd.read_csv(lazada_path).rename(columns={'final_text': 'review'})
    df_synthetic = pd.read_csv(synthetic_path)
    
    # 2. CROSS-CHECK (Synthetic vs Lazada)
    print("Loading LaBSE model...")
    model = SentenceTransformer('sentence-transformers/LaBSE')
    
    print("Encoding datasets...")
    lazada_emb = model.encode(df_lazada['review'].tolist(), show_progress_bar=True)
    synthetic_emb = model.encode(df_synthetic['review'].tolist(), show_progress_bar=True)
    
    print("Computing Cross-Similarity...")
    cross_sim = cosine_similarity(synthetic_emb, lazada_emb)
    
    redundant_indices = []
    cross_report = []
    
    print("Filtering synthetic data...")
    for i in tqdm(range(len(cross_sim)), desc="Cross-Check"):
        max_score = np.max(cross_sim[i])
        if max_score > 0.85:
            redundant_indices.append(i)
            if len(cross_report) < 50:
                laz_idx = np.argmax(cross_sim[i])
                cross_report.append({
                    "Similarity": f"{max_score:.4f}",
                    "Real Lazada Review": df_lazada.iloc[laz_idx]['review'],
                    "Redundant Synthetic Review": df_synthetic.iloc[i]['review']
                })

    # 3. MERGE
    df_synthetic_novel = df_synthetic.drop(redundant_indices)
    
    if cross_report:
        pd.DataFrame(cross_report).to_csv(os.path.join(REPORT_DIR, "cross_dataset_redundancy_appendix.csv"), index=False)

    df_hybrid = pd.concat([df_lazada, df_synthetic_novel], ignore_index=True)
    df_hybrid = df_hybrid.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save CSV and TXT
    csv_path = os.path.join(PROCESSED_DIR, "hybrid_corpus.csv")
    txt_path = os.path.join(PROCESSED_DIR, "hybrid_corpus.txt")
    
    df_hybrid.to_csv(csv_path, index=False)
    
    with open(txt_path, 'w', encoding='utf-8') as f:
        for line in df_hybrid['review']:
            f.write(str(line) + "\n")
            
    print(f"SUCCESS: Hybrid Corpus saved to {csv_path}")

if __name__ == "__main__":
    main()