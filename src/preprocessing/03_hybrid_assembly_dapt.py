"""
Cross-Dataset Deduplication and Hybrid Corpus Generation Script.

This script merges cleaned real Lazada data and synthetic data.
It evaluates semantic similarity across datasets using the LaBSE model
to identify and remove redundant synthetic reviews. The resulting
hybrid corpus is shuffled and exported as both CSV and TXT files.
"""

import os
import sys

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ==============================================================================
# PATH CONFIGURATION
# ==============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(SCRIPT_DIR)
BASE_DIR = os.path.dirname(SRC_DIR)

INTERIM_DIR = os.path.join(BASE_DIR, 'data', '02_interim')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', '03_processed')
REPORT_DIR = os.path.join(INTERIM_DIR, 'dedup_reports')

# Create necessary output directories
os.makedirs(PROCESSED_DIR, exist_ok=True)


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def main():
    """
    Executes the cross-dataset deduplication and merging pipeline.

    This function performs three main operations:
    1. Loads the pre-cleaned Lazada and synthetic datasets.
    2. Encodes the texts and computes cross-dataset similarity to filter
       synthetic reviews that duplicate existing real reviews.
    3. Merges the novel synthetic data with the real data, shuffles the 
       combined corpus, and exports it.
    """
    print(f"Project Root Detected: {BASE_DIR}")

    # --------------------------------------------------------------------------
    # 1. LOAD DATASETS
    # --------------------------------------------------------------------------
    lazada_path = os.path.join(INTERIM_DIR, "cleaned_lazada_data.csv")
    synthetic_path = os.path.join(INTERIM_DIR, "cleaned_synthetic_data.csv")
    
    # Verify the existence of intermediate files
    if not (os.path.exists(lazada_path) and os.path.exists(synthetic_path)):
        print("[ERROR] Intermediate files missing. Run scripts 01 and 02.")
        return

    print("Loading intermediate datasets...")
    # Load datasets and standardize the column name for the text data
    df_lazada = pd.read_csv(lazada_path).rename(columns={'final_text': 'review'})
    df_synthetic = pd.read_csv(synthetic_path)
    
    # --------------------------------------------------------------------------
    # 2. CROSS-CHECK (Synthetic vs Lazada)
    # --------------------------------------------------------------------------
    print("Loading LaBSE model...")
    model = SentenceTransformer('sentence-transformers/LaBSE')
    
    print("Encoding datasets into embeddings...")
    lazada_emb = model.encode(df_lazada['review'].tolist(), show_progress_bar=True)
    synthetic_emb = model.encode(df_synthetic['review'].tolist(), show_progress_bar=True)
    
    print("Computing Cross-Similarity...")
    # Calculate cosine similarity between synthetic embeddings and real embeddings
    cross_sim = cosine_similarity(synthetic_emb, lazada_emb)
    
    redundant_indices = []
    cross_report = []
    
    print("Filtering synthetic data...")
    # Identify synthetic reviews that are semantically identical to real reviews
    for i in tqdm(range(len(cross_sim)), desc="Cross-Check"):
        max_score = np.max(cross_sim[i])
        
        # Mark index for removal if maximum similarity exceeds the threshold
        if max_score > 0.85:
            redundant_indices.append(i)
            
            # Limit the sample report to 50 entries
            if len(cross_report) < 50:
                laz_idx = np.argmax(cross_sim[i])
                cross_report.append({
                    "Similarity": f"{max_score:.4f}",
                    "Real Lazada Review": df_lazada.iloc[laz_idx]['review'],
                    "Redundant Synthetic Review": df_synthetic.iloc[i]['review']
                })

    # --------------------------------------------------------------------------
    # 3. MERGE AND SAVE
    # --------------------------------------------------------------------------
    # Remove redundant entries from the synthetic dataset
    df_synthetic_novel = df_synthetic.drop(redundant_indices)
    
    # Export a sample report of redundant entries for manual review
    if cross_report:
        report_file = os.path.join(REPORT_DIR, "cross_dataset_redundancy_appendix.csv")
        pd.DataFrame(cross_report).to_csv(report_file, index=False)

    # Concatenate the novel synthetic data with the real Lazada data
    df_hybrid = pd.concat([df_lazada, df_synthetic_novel], ignore_index=True)
    
    # Shuffle the hybrid corpus completely and reset the index
    df_hybrid = df_hybrid.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Define output paths
    csv_path = os.path.join(PROCESSED_DIR, "hybrid_corpus.csv")
    txt_path = os.path.join(PROCESSED_DIR, "hybrid_corpus.txt")
    
    # Save the dataframe as CSV
    df_hybrid.to_csv(csv_path, index=False)
    
    # Save the dataframe as plain text
    with open(txt_path, 'w', encoding='utf-8') as f:
        for line in df_hybrid['review']:
            f.write(str(line) + "\n")
            
    print(f"SUCCESS: Hybrid Corpus saved to {csv_path}")


if __name__ == "__main__":
    main()