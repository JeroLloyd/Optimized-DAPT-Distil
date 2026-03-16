import os

# --- 1. DIRECTORY STRUCTURE ---
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)

# Corpus replaces Data
CORPUS_DIR = os.path.join(PROJECT_ROOT, 'corpus')
RAW_DIR = os.path.join(CORPUS_DIR, '01_raw')
INTERIM_DIR = os.path.join(CORPUS_DIR, '02_interim')
PROCESSED_DIR = os.path.join(CORPUS_DIR, '03_processed')
FIRECS_DIR = os.path.join(PROCESSED_DIR, 'FiReCS_Final')

# Artifacts replaces Models
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, 'artifacts')
MODEL_A_DIR = os.path.join(ARTIFACTS_DIR, "model_a_base")
MODEL_B_DIR = os.path.join(ARTIFACTS_DIR, "model_b_dapt")
MODEL_C_DIR = os.path.join(ARTIFACTS_DIR, "model_c_xlmr")
MODEL_D_DIR = os.path.join(ARTIFACTS_DIR, "model_d_onnx")

# Reports
REPORTS_DIR = os.path.join(PROJECT_ROOT, 'reports')
FIGURES_DIR = os.path.join(REPORTS_DIR, 'figures')
METRICS_DIR = os.path.join(REPORTS_DIR, 'metrics')

# Auto-generate folder structure
for path in [RAW_DIR, INTERIM_DIR, PROCESSED_DIR, FIRECS_DIR, ARTIFACTS_DIR, FIGURES_DIR, METRICS_DIR]:
    os.makedirs(path, exist_ok=True)

# --- 2. REPRODUCIBILITY & SOURCES ---
SEED = 42
URL_LAZADA_REVIEWS_JSON = "https://raw.githubusercontent.com/EricEchemane/Filipino-Tagalog-Product-Reviews-Sentiment-Analysis/main/data/reviews.json"
HF_FIRECS = "ccosme/FiReCS"

# --- 3. HYPERPARAMETERS ---
MAX_LEN = 128
BATCH_SIZE = 16 
LEARNING_RATE = 2e-5
EPOCHS_DAPT = 30 
EPOCHS_FINETUNE = 5
MLM_PROBABILITY = 0.15