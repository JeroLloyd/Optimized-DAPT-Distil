import pandas as pd
import numpy as np
import json
import re
import os
import sys
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- PATH CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
RAW_DIR = os.path.join(BASE_DIR, 'data', '01_raw')
INTERIM_DIR = os.path.join(BASE_DIR, 'data', '02_interim')
REPORT_DIR = os.path.join(INTERIM_DIR, 'dedup_reports')

os.makedirs(INTERIM_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# --- HELPER FUNCTIONS (From Notebook 01) ---
def is_gibberish(text):
    """Detects strings with non-standard vowel ratios or excessive word lengths."""
    if not isinstance(text, str): return True
    words = text.split()
    if any(len(word) > 15 for word in words): 
        return True
    
    vowels = len(re.findall(r'[aeiou]', text.lower()))
    total_chars = len(re.sub(r'[^a-zA-Z]', '', text))
    
    if total_chars == 0: return True
    ratio = vowels / total_chars
    return ratio < 0.2 or ratio > 0.9

def clean_text(text):
    """Basic normalization: lowercase, remove URLs, filter non-alphanumeric."""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def main():
    print(f"Project Root Detected: {BASE_DIR}")
    
    # 1. LOAD DATA
    # Check if the user provided the manually filtered English-removed file
    manual_path = os.path.join(INTERIM_DIR, "lazada_merged_corpus_english_removed.csv")
    
    if os.path.exists(manual_path):
        print(f"Loading manually filtered file: {manual_path}")
        df = pd.read_csv(manual_path)
        # Ensure we have the 'final_text' column
        if 'final_text' not in df.columns:
            # If the user saved it with just 'text' or different name, try to fix
            if 'text' in df.columns:
                 df['final_text'] = df['text']
            else:
                 # Assume first column is the text
                 df['final_text'] = df.iloc[:, 0]
    else:
        print("Manual English-removed file not found. Starting from RAW files...")
        # Fallback to Notebook 01 Raw Merge Logic
        qa_path = os.path.join(RAW_DIR, "lazada-qa-taglish.csv")
        review_path = os.path.join(RAW_DIR, "lazada-review-filipino.json")
        
        if not os.path.exists(qa_path) or not os.path.exists(review_path):
            print(f"[ERROR] Raw files missing in {RAW_DIR}")
            return

        # Load QA
        try:
            df_qa = pd.read_csv(qa_path)
            if 'question' in df_qa.columns:
                 df_qa = df_qa[['question']].rename(columns={'question': 'text'})
            elif 'text' in df_qa.columns:
                 df_qa = df_qa[['text']]
        except:
             df_qa = pd.DataFrame(columns=['text'])

        # Load Reviews
        try:
            df_review = pd.read_json(review_path)
            # Dynamic column check
            target_col = None
            for col in ['reviewContent', 'review', 'content', 'text']:
                if col in df_review.columns:
                    target_col = col
                    break
            
            if target_col:
                df_review = df_review[[target_col]].rename(columns={target_col: 'text'})
            else:
                # Try lines=True
                df_review = pd.read_json(review_path, lines=True)
                df_review = df_review.rename(columns={df_review.columns[0]: 'text'})
        except:
             df_review = pd.DataFrame(columns=['text'])
        
        df = pd.concat([df_qa, df_review], ignore_index=True)
        
        # Apply Cleaning
        print("Applying cleaning rules...")
        df['final_text'] = df['text'].apply(clean_text)
        df = df[df['final_text'].apply(lambda x: len(str(x).split()) >= 3)]
        df = df[~df['final_text'].apply(is_gibberish)]

    print(f"Data ready for deduplication. Count: {len(df)}")

    # 2. SEMANTIC DEDUPLICATION (LaBSE > 0.85)
    print("Loading LaBSE model...")
    model = SentenceTransformer('sentence-transformers/LaBSE')
    
    print("Encoding...")
    embeddings = model.encode(df['final_text'].tolist(), show_progress_bar=True)
    
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
                        "Original Sentence (Keep)": df.iloc[i]['final_text'],
                        "Duplicate Sentence (Remove)": df.iloc[j]['final_text']
                    })

    # 3. SAVE RESULTS
    if report_data:
        report_df = pd.DataFrame(report_data)
        report_path = os.path.join(REPORT_DIR, "lazada_deduplication_appendix.csv")
        report_df.to_csv(report_path, index=False)
        print(f"Report saved: {report_path}")

    df_final = df.iloc[[i for i in range(len(df)) if i not in to_drop]].reset_index(drop=True)
    output_path = os.path.join(INTERIM_DIR, "cleaned_lazada_data.csv")
    df_final.to_csv(output_path, index=False)
    
    print(f"SUCCESS: Saved cleaned Lazada corpus to {output_path}")

if __name__ == "__main__":
    main()