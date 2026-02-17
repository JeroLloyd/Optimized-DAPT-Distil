import pandas as pd
import numpy as np
import os
import re
import sys
from sklearn.model_selection import train_test_split

# --- PATH CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
# Raw files are expected in data/01_raw/FireCS/
RAW_DIR = os.path.join(BASE_DIR, 'data', '01_raw', 'FireCS')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', '03_processed', 'FiReCS_Final')

os.makedirs(PROCESSED_DIR, exist_ok=True)
SEED = 42

# --- HELPER FUNCTIONS (From Notebook 04) ---
def basic_clean(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def is_word_gibberish(word):
    if len(word) > 15: return True
    vowels = len(re.findall(r'[aeiou]', word))
    if len(word) > 0:
        ratio = vowels / len(word)
        if ratio < 0.2 or ratio > 0.9: return True
    return False

def main():
    print(f"Project Root Detected: {BASE_DIR}")

    # 1. LOAD RAW DATA
    # Updated to match your file names: firecs_train_set.csv and firecs_test_set.csv
    train_path = os.path.join(RAW_DIR, "firecs_train_set.csv")
    test_path = os.path.join(RAW_DIR, "firecs_test_set.csv")

    print(f"Loading files from: {RAW_DIR}")
    try:
        if not os.path.exists(train_path) or not os.path.exists(test_path):
            print(f"[ERROR] One or both files missing:")
            print(f"  Expected: {train_path}")
            print(f"  Expected: {test_path}")
            # Try fallback names just in case
            print("  Trying fallback names...")
            train_path = os.path.join(RAW_DIR, "train_data_firecs.csv")
            test_path = os.path.join(RAW_DIR, "test_data_firecs.csv")
            
        train_raw = pd.read_csv(train_path)
        test_raw = pd.read_csv(test_path)
        
        # Merge for unified processing
        df_raw = pd.concat([train_raw, test_raw], ignore_index=True)
        print(f"Total raw rows loaded: {len(df_raw)}")
        
    except FileNotFoundError:
        print(f"[ERROR] FireCS raw files not found in {RAW_DIR}")
        return
    except Exception as e:
        print(f"[ERROR] Failed to load CSVs: {e}")
        return

    # 2. GIBBERISH SALVAGING
    print("Applying Gibberish Salvaging...")
    final_rows = []
    
    for _, row in df_raw.iterrows():
        # Ensure 'review' column exists (handle varied case)
        if 'review' not in row and 'text' in row:
            review_text = row['text']
        elif 'review' in row:
            review_text = row['review']
        else:
            continue # Skip if no text found
            
        text = basic_clean(review_text)
        words = text.split()
        
        # Salvage logic: Keep valid words from mixed sentences
        if any(is_word_gibberish(w) for w in words):
            clean_words = [w for w in words if not is_word_gibberish(w)]
            if len(clean_words) >= 3:
                row['review'] = " ".join(clean_words)
                final_rows.append(row)
        else:
            if len(words) >= 3:
                row['review'] = text # Update with cleaned text
                final_rows.append(row)
    
    if not final_rows:
        print("[ERROR] No rows remained after cleaning! Check input data format.")
        return

    df_cleaned = pd.DataFrame(final_rows)
    
    # Remove duplicates
    prev_len = len(df_cleaned)
    df_cleaned = df_cleaned.drop_duplicates(subset=['review'], keep='first')
    
    print(f"Cleaned count: {len(df_cleaned)} (Dropped {prev_len - len(df_cleaned)} duplicates)")

    # 3. STRATIFIED SPLIT (80/10/10)
    print("Splitting 80/10/10...")
    
    try:
        # Split 1: 80% Train, 20% Temp
        train_df, temp_df = train_test_split(
            df_cleaned, test_size=0.20, random_state=SEED, stratify=df_cleaned['label']
        )
        
        # Split 2: 50% of Temp -> Val (10%), Test (10%)
        val_df, test_df = train_test_split(
            temp_df, test_size=0.50, random_state=SEED, stratify=temp_df['label']
        )
        
        # Save
        train_df.to_csv(os.path.join(PROCESSED_DIR, "train.csv"), index=False)
        val_df.to_csv(os.path.join(PROCESSED_DIR, "val.csv"), index=False)
        test_df.to_csv(os.path.join(PROCESSED_DIR, "test.csv"), index=False)
        
        print(f"SUCCESS: Splits saved to {PROCESSED_DIR}")
        print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
    except ValueError as e:
        print(f"[ERROR] Stratification failed: {e}")
        print("Check if 'label' column exists and has enough samples per class.")

if __name__ == "__main__":
    main()