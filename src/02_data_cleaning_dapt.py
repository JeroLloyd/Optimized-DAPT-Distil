import pandas as pd
import re
import sys
import os
import json

# Import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

# --- YOUR CUSTOM CLEANING LOGIC ---
def is_gibberish(text):
    """
    Heuristic function to identify gibberish.
    Criteria: 
    1. Any word > 15 chars.
    2. Vowel ratio < 0.2 or > 0.8 (for words > 5 chars).
    """
    words = text.split()
    if any(len(word) > 15 for word in words):
        return True
    
    # Remove non-alpha for vowel calculation
    clean_text = re.sub(r'[^a-zA-Z]', '', text)
    if len(clean_text) > 5:
        vowels = len(re.findall(r'[aeiouAEIOU]', clean_text))
        total_chars = len(clean_text)
        vowel_ratio = vowels / total_chars
        if vowel_ratio < 0.2 or vowel_ratio > 0.8:
            return True
    return False

def basic_clean(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-z0-9\s]', '', text) 
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def process_gibberish_entry(text):
    """
    Salvage logic: Keep non-gibberish words. 
    If result < 3 words, discard entire entry.
    """
    if is_gibberish(text):
        # Try to keep words that are NOT too long
        salvaged = ' '.join([w for w in text.split() if len(w) <= 15])
        if len(salvaged.split()) < 3:
            return None # Discard
        else:
            return salvaged # Keep salvaged
    return text

def run_cleaning():
    print("--- STARTING DAPT DATA CLEANING ---")
    
    if not os.path.exists(config.RAW_LAZADA_QA_PATH):
        print("[ERROR] Run '01_data_ingestion.py' first.")
        return

    # 1. Load LazadaQA (CSV)
    print("Loading LazadaQA (CSV)...")
    try:
        df_qa = pd.read_csv(config.RAW_LAZADA_QA_PATH)
    except Exception as e:
        print(f"[ERROR] Could not read LazadaQA CSV: {e}")
        return

    # 2. Load Lazada Reviews (JSON) -> UPDATED
    print("Loading Lazada Reviews (JSON)...")
    try:
        with open(config.RAW_LAZADA_REVIEWS_PATH, 'r', encoding='utf-8') as f:
            reviews_data = json.load(f)
        df_reviews = pd.DataFrame(reviews_data)
    except Exception as e:
        print(f"[ERROR] Could not read Lazada Reviews JSON: {e}")
        return

    # 3. Extract specific columns
    print("Extracting text columns...")
    
    # QA Column
    qa_col = 'question' if 'question' in df_qa.columns else df_qa.columns[0]
    
    # Reviews Column (Check for 'review', 'text', 'content', etc.)
    possible_cols = ['review', 'text', 'review_text', 'content']
    review_col = next((c for c in possible_cols if c in df_reviews.columns), df_reviews.columns[0])
    
    print(f"   QA Column: '{qa_col}' | Review Column: '{review_col}'")

    lazada_questions = df_qa[qa_col].astype(str)
    lazada_reviews = df_reviews[review_col].astype(str)

    # 4. Merge (Concatenate)
    df = pd.DataFrame({'text': pd.concat([lazada_questions, lazada_reviews], ignore_index=True)})
    initial_count = len(df)

    # 5. Apply Cleaning Pipeline
    print("Applying Cleaning & Gibberish Detection...")
    
    # A. Drop NA/Empty
    df = df.dropna(subset=['text'])
    df = df[df['text'].str.strip() != ""]
    
    # B. Basic Clean
    df['cleaned'] = df['text'].apply(basic_clean)
    
    # C. Length Filter (< 3 words)
    df = df[df['cleaned'].apply(lambda x: len(x.split()) >= 3)]
    
    # D. Gibberish Processing
    df['final_text'] = df['cleaned'].apply(process_gibberish_entry)
    df = df.dropna(subset=['final_text']) # Remove discarded lines
    
    # E. Deduplication
    df_final = df.drop_duplicates(subset=['final_text'])

    # 6. Save
    os.makedirs(os.path.dirname(config.DAPT_CORPUS_PATH), exist_ok=True)
    with open(config.DAPT_CORPUS_PATH, 'w', encoding='utf-8') as f:
        for line in df_final['final_text']:
            f.write(line + '\n')
    
    print(f"Initial Count: {initial_count}")
    print(f"Final Count:   {len(df_final)}")
    print(f"[SUCCESS] DAPT Corpus saved to: {config.DAPT_CORPUS_PATH}")

if __name__ == "__main__":
    run_cleaning()