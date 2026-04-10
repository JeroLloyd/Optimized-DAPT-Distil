"""
Data Processing and Gibberish Salvaging Script for FireCS Dataset.

This script loads raw training and testing datasets. It cleans text, removes 
gibberish words, and filters duplicate entries. Finally, it performs a 
stratified 80/10/10 split for training, validation, and testing phases.
"""

import os
import re
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ==============================================================================
# PATH CONFIGURATION
# ==============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(SCRIPT_DIR)
BASE_DIR = os.path.dirname(SRC_DIR)

# Raw files are expected in data/01_raw/FireCS/
RAW_DIR = os.path.join(BASE_DIR, 'data', '01_raw', 'FireCS')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', '03_processed', 'FiReCS_Final')

os.makedirs(PROCESSED_DIR, exist_ok=True)
SEED = 42


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def basic_clean(text):
    """
    Cleans raw text by normalizing case, removing URLs, and filtering special characters.

    Args:
        text (str): The raw input string.

    Returns:
        str: The cleaned and normalized string.
    """
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def is_word_gibberish(word):
    """
    Evaluates a single word to determine if it is gibberish.
    It flags words exceeding 15 characters or those with abnormal vowel ratios.

    Args:
        word (str): The specific word to evaluate.

    Returns:
        bool: True if the word is classified as gibberish. False otherwise.
    """
    if len(word) > 15: 
        return True
        
    vowels = len(re.findall(r'[aeiou]', word))
    
    if len(word) > 0:
        ratio = vowels / len(word)
        if ratio < 0.2 or ratio > 0.9: 
            return True
            
    return False


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def main():
    """
    Executes the FireCS data preparation pipeline.
    
    This function handles data loading, applies text cleaning and gibberish 
    salvaging rules, removes duplicates, and generates stratified dataset splits.
    """
    print(f"Project Root Detected: {BASE_DIR}")

    # --------------------------------------------------------------------------
    # 1. LOAD RAW DATA
    # --------------------------------------------------------------------------
    train_path = os.path.join(RAW_DIR, "firecs_train_set.csv")
    test_path = os.path.join(RAW_DIR, "firecs_test_set.csv")

    print(f"Loading files from: {RAW_DIR}")
    
    try:
        # Check for primary file names and apply fallback names if missing
        if not os.path.exists(train_path) or not os.path.exists(test_path):
            print("[ERROR] One or both files missing:")
            print(f"  Expected: {train_path}")
            print(f"  Expected: {test_path}")
            print("  Trying fallback names...")
            train_path = os.path.join(RAW_DIR, "train_data_firecs.csv")
            test_path = os.path.join(RAW_DIR, "test_data_firecs.csv")
            
        train_raw = pd.read_csv(train_path)
        test_raw = pd.read_csv(test_path)
        
        # Merge datasets for unified processing
        df_raw = pd.concat([train_raw, test_raw], ignore_index=True)
        print(f"Total raw rows loaded: {len(df_raw)}")
        
    except FileNotFoundError:
        print(f"[ERROR] FireCS raw files not found in {RAW_DIR}")
        return
    except Exception as e:
        print(f"[ERROR] Failed to load CSVs: {e}")
        return

    # --------------------------------------------------------------------------
    # 2. GIBBERISH SALVAGING & CLEANING
    # --------------------------------------------------------------------------
    print("Applying Gibberish Salvaging...")
    final_rows = []
    
    for _, row in df_raw.iterrows():
        # Ensure the 'review' column exists or standardize from 'text'
        if 'review' not in row and 'text' in row:
            review_text = row['text']
        elif 'review' in row:
            review_text = row['review']
        else:
            continue
            
        text = basic_clean(review_text)
        words = text.split()
        
        # Salvage logic: Retain valid words from mixed sentences
        if any(is_word_gibberish(w) for w in words):
            clean_words = [w for w in words if not is_word_gibberish(w)]
            if len(clean_words) >= 3:
                row['review'] = " ".join(clean_words)
                final_rows.append(row)
        else:
            # Keep the cleaned text if it meets the minimum length requirement
            if len(words) >= 3:
                row['review'] = text
                final_rows.append(row)
                
    if not final_rows:
        print("[ERROR] No rows remained after cleaning! Check input data format.")
        return

    df_cleaned = pd.DataFrame(final_rows)
    
    # Remove duplicate entries based on the cleaned review text
    prev_len = len(df_cleaned)
    df_cleaned = df_cleaned.drop_duplicates(subset=['review'], keep='first')
    
    print(f"Cleaned count: {len(df_cleaned)} (Dropped {prev_len - len(df_cleaned)} duplicates)")

    # --------------------------------------------------------------------------
    # 3. STRATIFIED SPLIT (80/10/10)
    # --------------------------------------------------------------------------
    print("Splitting datasets using an 80/10/10 ratio...")
    
    try:
        # Split 1: Isolate 80% for training and reserve 20% for testing pools
        train_df, temp_df = train_test_split(
            df_cleaned, test_size=0.20, random_state=SEED, stratify=df_cleaned['label']
        )
        
        # Split 2: Divide the reserved pool evenly into validation and test sets
        val_df, test_df = train_test_split(
            temp_df, test_size=0.50, random_state=SEED, stratify=temp_df['label']
        )
        
        # Export the final dataset splits
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