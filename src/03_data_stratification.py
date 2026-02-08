import pandas as pd
import re
import sys
import os
from sklearn.model_selection import train_test_split
from datasets import load_dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

# --- YOUR CUSTOM LOGIC (from firecs notebook) ---
def basic_clean(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-z0-9\s]', '', text) 
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def is_word_gibberish(word):
    if len(word) > 15: return True
    vowels = len(re.findall(r'[aeiou]', word))
    if len(word) > 3:
        ratio = vowels / len(word)
        if ratio < 0.20 or ratio > 0.80:
            return True
    return False

def clean_firecs_row(text):
    """
    Salvage attempt: keep only non-gibberish words.
    """
    words = text.split()
    if any(is_word_gibberish(w) for w in words):
        # Keep only clean words
        clean_words = [w for w in words if not is_word_gibberish(w)]
        if len(clean_words) >= 3:
            return " ".join(clean_words)
        else:
            return None # Discard (Too short after cleaning)
    else:
        # No gibberish, check length
        if len(words) >= 3:
            return text
        else:
            return None # Discard (Too short)

def run_stratification():
    print("--- STARTING FIRECS STRATIFICATION (With Cleaning) ---")
    config.set_seed() # CRITICAL: SEED = 42

    # 1. Load Data (Merge Train+Test from Raw Download)
    if not os.path.exists(config.RAW_FIRECS_PATH):
        print("[ERROR] Run 'data_ingestion.py' first.")
        return

    df = pd.read_csv(config.RAW_FIRECS_PATH)
    initial_len = len(df)
    
    # 2. Pipeline: Basic Clean -> Gibberish Filter -> Deduplicate
    print("Running Cleaning Pipeline...")
    
    # A. Basic Clean
    df['review'] = df['review'].apply(basic_clean)
    
    # B. Gibberish/Length Filter
    # We apply the logic row-by-row
    clean_rows = []
    for _, row in df.iterrows():
        cleaned_val = clean_firecs_row(row['review'])
        if cleaned_val:
            row['review'] = cleaned_val
            clean_rows.append(row)
            
    df_cleaned = pd.DataFrame(clean_rows)
    
    # C. Deduplication
    df_cleaned = df_cleaned.drop_duplicates(subset=['review'], keep='first')
    
    print(f"Cleaned Count: {len(df_cleaned)} (Dropped {initial_len - len(df_cleaned)})")

    # 3. Stratified Split (80 / 10 / 10)
    # Using 'label' column for stratification
    print("Splitting Data (Seed=42)...")
    
    # Split 1: 80% Train, 20% Temp
    train_df, temp_df = train_test_split(
        df_cleaned, 
        test_size=0.20, 
        random_state=config.SEED, 
        stratify=df_cleaned['label']
    )
    
    # Split 2: 50% Val, 50% Test (from the 20% Temp)
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=0.50, 
        random_state=config.SEED, 
        stratify=temp_df['label']
    )

    # 4. Save to Processed Folder
    os.makedirs(os.path.dirname(config.TRAIN_PATH), exist_ok=True)
    train_df.to_csv(config.TRAIN_PATH, index=False)
    val_df.to_csv(config.VAL_PATH, index=False)
    test_df.to_csv(config.TEST_PATH, index=False)

    print(f"[SUCCESS] Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

if __name__ == "__main__":
    run_stratification()