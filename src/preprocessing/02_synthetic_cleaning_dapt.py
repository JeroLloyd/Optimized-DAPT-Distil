"""
Semantic Deduplication Script for Synthetic Data.

This script processes raw synthetic data and applies semantic deduplication. 
It uses the LaBSE sentence transformer model to encode text into embeddings. 
The script then computes a cosine similarity matrix to identify and remove 
duplicate entries based on a defined threshold. Finally, it exports the 
cleaned dataset and a deduplication report.
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

RAW_DIR = os.path.join(BASE_DIR, 'data', '01_raw')
INTERIM_DIR = os.path.join(BASE_DIR, 'data', '02_interim')
REPORT_DIR = os.path.join(INTERIM_DIR, 'dedup_reports')

# Create necessary directories if they do not exist
os.makedirs(INTERIM_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def main():
    """
    Executes the semantic deduplication pipeline.

    The function performs three primary operations:
    1. Loads the raw synthetic dataset.
    2. Encodes text and filters out semantically similar duplicates.
    3. Saves the cleaned data and a report of the removed duplicates.
    """
    print(f"Project Root Detected: {BASE_DIR}")
    
    # --------------------------------------------------------------------------
    # 1. LOAD DATA
    # --------------------------------------------------------------------------
    input_path = os.path.join(RAW_DIR, "raw_synthetic_data.csv")
    
    if not os.path.exists(input_path):
        print(f"[ERROR] Missing file at {input_path}")
        return

    print(f"Loading data from: {input_path}")
    df = pd.read_csv(input_path)
    
    # --------------------------------------------------------------------------
    # 2. SEMANTIC DEDUPLICATION
    # --------------------------------------------------------------------------
    print("Loading LaBSE model...")
    model = SentenceTransformer('sentence-transformers/LaBSE')
    
    print("Encoding text into embeddings...")
    # Generate vector representations for each review
    embeddings = model.encode(df['review'].tolist(), show_progress_bar=True)
    
    print("Computing Similarity Matrix...")
    # Calculate pairwise cosine similarity across all embeddings
    sim_matrix = cosine_similarity(embeddings)
    
    threshold = 0.85
    to_drop = set()
    report_data = []

    print("Identifying duplicates...")
    # Iterate through the upper triangle of the similarity matrix
    for i in tqdm(range(len(sim_matrix)), desc="Deduplication"):
        # Skip rows already flagged for removal
        if i in to_drop:
            continue
            
        for j in range(i + 1, len(sim_matrix)):
            score = sim_matrix[i][j]
            
            # Flag the duplicate index if similarity exceeds the threshold
            if score > threshold:
                to_drop.add(j)
                
                # Log non-exact duplicates for the appendix report
                if score < 0.99:
                    report_data.append({
                        "Similarity Score": f"{score:.4f}",
                        "Original Sentence (Keep)": df.iloc[i]['review'],
                        "Duplicate Sentence (Remove)": df.iloc[j]['review']
                    })

    # --------------------------------------------------------------------------
    # 3. SAVE RESULTS
    # --------------------------------------------------------------------------
    # Export the deduplication appendix if any partial duplicates were found
    if report_data:
        report_df = pd.DataFrame(report_data)
        report_path = os.path.join(REPORT_DIR, "synthetic_deduplication_appendix.csv")
        report_df.to_csv(report_path, index=False)
        print(f"Report saved: {report_path}")

    # Generate the final dataframe by excluding dropped indices
    df_final = df.iloc[[i for i in range(len(df)) if i not in to_drop]].reset_index(drop=True)
    
    output_path = os.path.join(INTERIM_DIR, "cleaned_synthetic_data.csv")
    df_final.to_csv(output_path, index=False)
    
    print(f"SUCCESS: Saved cleaned Synthetic corpus to {output_path}")


if __name__ == "__main__":
    main()