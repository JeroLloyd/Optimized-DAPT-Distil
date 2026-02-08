import pandas as pd
from datasets import load_dataset
import os
import sys
import requests

# Import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

def download_file(url, save_path):
    print(f"Downloading from {url}...")
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print(f"[SUCCESS] Saved to {save_path}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to download {url}: {e}")
        return False

def ingest_data():
    print("--- STARTING DATA INGESTION ---")
    os.makedirs(os.path.dirname(config.RAW_LAZADA_QA_PATH), exist_ok=True)

    # 1. LazadaQA (Manual Local Check)
    if os.path.exists(config.RAW_LAZADA_QA_PATH):
        print(f"[SUCCESS] Found LazadaQA file at: {config.RAW_LAZADA_QA_PATH}")
    else:
        print(f"[ERROR] LazadaQA file NOT found!")
        print(f"Please move 'LazadaQA-Taglish-7k.csv' to: {config.RAW_LAZADA_QA_PATH}")
        sys.exit(1) # Stop execution if file is missing

    # 2. Lazada Reviews (UPDATED: Download JSON from GitHub)
    if not os.path.exists(config.RAW_LAZADA_REVIEWS_PATH):
        print(f"Downloading Lazada Reviews (JSON)...")
        # Uses the new URL_LAZADA_REVIEWS_JSON from config.py
        success = download_file(config.URL_LAZADA_REVIEWS_JSON, config.RAW_LAZADA_REVIEWS_PATH)
        if not success:
            sys.exit(1)
    else:
        print(f"[INFO] Lazada Reviews already exists at {config.RAW_LAZADA_REVIEWS_PATH}")

    # 3. FiReCS (Hugging Face)
    if not os.path.exists(config.RAW_FIRECS_PATH):
        print("Downloading FiReCS from Hugging Face...")
        try:
            # We add trust_remote_code here too just in case
            dataset = load_dataset(config.HF_FIRECS, trust_remote_code=True)
            
            df_train = pd.DataFrame(dataset['train'])
            df_test = pd.DataFrame(dataset['test'])
            df_full = pd.concat([df_train, df_test])
            df_full.to_csv(config.RAW_FIRECS_PATH, index=False)
            print(f"[SUCCESS] Saved {len(df_full)} rows (Combined FiReCS).")
        except Exception as e:
            print(f"[ERROR] FiReCS download failed: {e}")
            sys.exit(1)
    else:
        print("[INFO] FiReCS already exists.")

    print("\n--- DATA INGESTION COMPLETE ---")

if __name__ == "__main__":
    ingest_data()