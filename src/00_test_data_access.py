import sys
import os
import pandas as pd
from datasets import load_dataset
import requests

# Add parent directory to path so we can import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

def test_data_access():
    print("========================================================")
    print("      DIAGNOSTIC TEST: DATASET ACCESSIBILITY")
    print("========================================================")

    # --- TEST 1: LAZADA QA (Local File Check) ---
    print(f"\n[1/3] Testing LazadaQA (Local File)...")
    print(f"      Target Path: {config.RAW_LAZADA_QA_PATH}")
    
    if os.path.exists(config.RAW_LAZADA_QA_PATH):
        try:
            # Try to read the first 5 rows to ensure it's a valid CSV
            df = pd.read_csv(config.RAW_LAZADA_QA_PATH, nrows=5)
            print("      [SUCCESS] Local file found and readable.")
            print(f"      Columns Detected: {list(df.columns)}")
        except Exception as e:
            print(f"      [FAILED] File exists but cannot be read. Error: {e}")
    else:
        print("      [FAILED] File NOT found.")
        print("      ACTION REQUIRED: Move 'LazadaQA-Taglish-7k.csv' to 'data/raw/'")

    # --- TEST 2: LAZADA REVIEWS (GitHub JSON Check) ---
    # UPDATED: We now ping the GitHub URL instead of Hugging Face
    print(f"\n[2/3] Testing Lazada Reviews (GitHub JSON)...")
    print(f"      URL: {config.URL_LAZADA_REVIEWS_JSON}")
    try:
        # We stream=True to avoid downloading the whole file, just check headers
        response = requests.get(config.URL_LAZADA_REVIEWS_JSON, stream=True)
        if response.status_code == 200:
            print("      [SUCCESS] GitHub JSON link is valid and reachable.")
        else:
            print(f"      [FAILED] URL returned error code: {response.status_code}")
    except Exception as e:
        print(f"      [FAILED] Could not connect to GitHub. Error: {e}")

    # --- TEST 3: FiReCS (Hugging Face) ---
    print(f"\n[3/3] Testing FiReCS (Hugging Face)...")
    print(f"      ID: {config.HF_FIRECS}")
    try:
        # Added trust_remote_code=True as per your fix
        ds = load_dataset(config.HF_FIRECS, split='train', streaming=True, trust_remote_code=True)
        sample = next(iter(ds))
        print("      [SUCCESS] Connection established.")
        print(f"      Sample Review: \"{str(sample.get('review', 'N/A'))[:30]}...\"")
    except Exception as e:
        print(f"      [FAILED] Could not reach Hugging Face. Error: {e}")

    print("\n========================================================")
    print("      DIAGNOSTIC COMPLETE")
    print("========================================================")

if __name__ == "__main__":
    test_data_access()