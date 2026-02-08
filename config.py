import os
import torch
import random
import numpy as np

# --- 1. REPRODUCIBILITY SETUP (The "God Seed") ---
SEED = 42

def set_seed():
    """Sets the seed for reproducibility across all libraries."""
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(SEED)
    print(f"[INFO] Global Seed set to {SEED} for reproducibility.")

# --- 2. DATASET SOURCES ---

# NEW: Direct GitHub link for Lazada Reviews (JSON format)
URL_LAZADA_REVIEWS_JSON = "https://raw.githubusercontent.com/EricEchemane/Filipino-Tagalog-Product-Reviews-Sentiment-Analysis/main/data/reviews.json"

# Hugging Face Dataset IDs (Kept for reference, specifically FiReCS)
HF_FIRECS = "ccosme/FiReCS"

# --- 3. FILE PATHS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Raw Downloads
# UPDATE 1: Pointing to the specific file you downloaded manually
RAW_LAZADA_QA_PATH = os.path.join(DATA_DIR, "raw", "LazadaQA-Taglish-7k.csv")

# UPDATE 2: Changed filename extension to .json to match the new source
RAW_LAZADA_REVIEWS_PATH = os.path.join(DATA_DIR, "raw", "lazada_reviews.json")

# FiReCS remains the same
RAW_FIRECS_PATH = os.path.join(DATA_DIR, "raw", "firecs_full.csv")

RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# Processed Data
DAPT_CORPUS_PATH = os.path.join(DATA_DIR, "processed", "lazada_ecommerce_corpus.txt")
TRAIN_PATH = os.path.join(DATA_DIR, "processed", "train.csv")
VAL_PATH = os.path.join(DATA_DIR, "processed", "val.csv")
TEST_PATH = os.path.join(DATA_DIR, "processed", "test.csv")

# Model Output Directories
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_A_DIR = os.path.join(MODEL_DIR, "model_a_base")
MODEL_B_BASE_DIR = os.path.join(MODEL_DIR, "model_b_dapt_base")
MODEL_B_FINETUNED_DIR = os.path.join(MODEL_DIR, "model_b_dapt_finetuned")
MODEL_C_DIR = os.path.join(MODEL_DIR, "model_c_xlmr")
MODEL_D_DIR = os.path.join(MODEL_DIR, "model_d_optimized")

# --- 4. HYPERPARAMETERS ---
MAX_LEN = 128
BATCH_SIZE = 16 
LEARNING_RATE = 2e-5
EPOCHS_DAPT = 3
EPOCHS_FINETUNE = 5
MLM_PROBABILITY = 0.15