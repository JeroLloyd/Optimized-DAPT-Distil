import os
import sys
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

# --- AGGRESSIVE COMPATIBILITY PATCH ---
import transformers.utils
import transformers.modeling_utils
import transformers.models.auto
import transformers.utils.generic

# 1. Patch is_offline_mode
if not hasattr(transformers.utils, 'is_offline_mode'):
    transformers.utils.is_offline_mode = lambda: False

# 2. Patch get_parameter_dtype
if not hasattr(transformers.modeling_utils, 'get_parameter_dtype'):
    def _mock_get_parameter_dtype(model):
        try:
            return next(model.parameters()).dtype
        except Exception:
            return torch.float32
    transformers.modeling_utils.get_parameter_dtype = _mock_get_parameter_dtype

# 3. Patch AutoModelForVision2Seq
class MockAutoModelForVision2Seq:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        raise NotImplementedError("This is a mock class for compatibility.")
setattr(transformers, "AutoModelForVision2Seq", MockAutoModelForVision2Seq)
setattr(transformers.models.auto, "AutoModelForVision2Seq", MockAutoModelForVision2Seq)

# 4. Patch ONNX Exporter Requirements
if not hasattr(transformers.utils.generic, '_CAN_RECORD_REGISTRY'):
    transformers.utils.generic._CAN_RECORD_REGISTRY = {}
if not hasattr(transformers.utils.generic, 'OutputRecorder'):
    class MockOutputRecorder:
        pass
    transformers.utils.generic.OutputRecorder = MockOutputRecorder

from sklearn.metrics import confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- PATH CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Resolve root directory whether script is in root or src/
if os.path.basename(SCRIPT_DIR) == "src":
    BASE_DIR = os.path.dirname(SCRIPT_DIR)
else:
    BASE_DIR = SCRIPT_DIR

DATA_PATH = os.path.join(BASE_DIR, 'data', '03_processed', 'FiReCS_Final', 'test.csv')

# CHANGED: Now targeting Model A (Base) and Model B (DAPT)
MODEL_A_DIR = os.path.join(BASE_DIR, 'models', 'model_a_base')
MODEL_B_DIR = os.path.join(BASE_DIR, 'models', 'model_b_dapt')

REPORTS_DIR = os.path.join(BASE_DIR, 'reports', 'metrics')
FIGURES_DIR = os.path.join(BASE_DIR, 'reports', 'figures')

os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

LABEL_MAP = {0: "Negative", 1: "Neutral", 2: "Positive"}

def safe_load_tokenizer(model_path, fallback_name):
    """Bypasses PyPreTokenizerTypeWrapper fast tokenizer corruption."""
    try:
        return AutoTokenizer.from_pretrained(model_path)
    except Exception as e:
        print(f"  [WARN] Fast tokenizer failed: {e}. Retrying with use_fast=False...")
        try:
            return AutoTokenizer.from_pretrained(model_path, use_fast=False)
        except Exception as e2:
            print(f"  [WARN] Local tokenizer failed completely. Falling back to base {fallback_name}...")
            return AutoTokenizer.from_pretrained(fallback_name)

def main():
    print(f"Project Root Detected: {BASE_DIR}")
    print("--- GENERATING REAL DATA FOR CHAPTER 4.4 ---")

    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] Test data not found: {DATA_PATH}")
        return
    
    df = pd.read_csv(DATA_PATH)
    if 'review' in df.columns: 
        df = df.rename(columns={'review': 'text'})
        
    texts = df['text'].tolist()
    true_labels = df['label'].tolist()

    print("Loading Model A (Base DistilBERT)...")
    tokenizer_a = safe_load_tokenizer(MODEL_A_DIR, "distilbert-base-multilingual-cased")
    model_a = AutoModelForSequenceClassification.from_pretrained(MODEL_A_DIR)
    model_a.eval()

    print("Loading Model B (DAPT-DistilBERT)...")
    tokenizer_b = safe_load_tokenizer(MODEL_B_DIR, "distilbert-base-multilingual-cased")
    model_b = AutoModelForSequenceClassification.from_pretrained(MODEL_B_DIR)
    model_b.eval()

    def get_predictions(tokenizer, model, is_onnx=False):
        preds = []
        print(f"Predicting {len(texts)} sequences...")
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
            # Remove token_type_ids for distilbert compatibility if present
            if "token_type_ids" in inputs:
                inputs.pop("token_type_ids")

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                    
            pred = np.argmax(logits.cpu().numpy(), axis=1)[0]
            preds.append(pred)
        return preds

    print("\nRunning inferences for Base DistilBERT (Model A)...")
    preds_a = get_predictions(tokenizer_a, model_a, is_onnx=False)
    
    print("Running inferences for DAPT-DistilBERT (Model B)...")
    preds_b = get_predictions(tokenizer_b, model_b, is_onnx=False)

    df['Pred_Base'] = preds_a
    df['Pred_DAPT'] = preds_b

    print("\n--- CONFUSION MATRIX (Model B - DAPT) ---")
    cm = confusion_matrix(true_labels, preds_b, labels=[0, 1, 2])
    
    cm_df = pd.DataFrame(cm, 
                         index=["True Negative", "True Neutral", "True Positive"], 
                         columns=["Pred Negative", "Pred Neutral", "Pred Positive"])
    
    cm_csv_path = os.path.join(REPORTS_DIR, 'real_confusion_matrix.csv')
    cm_df.to_csv(cm_csv_path)
    print(cm_df)
    print(f"Saved Confusion Matrix CSV to: {cm_csv_path}")

    # --- PLOT CONFUSION MATRIX ---
    plt.figure(figsize=(8, 6))
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    plt.rcParams.update({'font.family': 'sans-serif'})

    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Negative", "Neutral", "Positive"],
                yticklabels=["Negative", "Neutral", "Positive"],
                annot_kws={"size": 14, "weight": "bold"},
                linewidths=1, linecolor='black')
    
    plt.title("Confusion Matrix: DAPT-DistilBERT (Model B)", fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Sentiment', fontsize=13, fontweight='bold')
    plt.xlabel('Predicted Sentiment', fontsize=13, fontweight='bold')
    plt.tight_layout()

    cm_fig_path = os.path.join(FIGURES_DIR, '7_confusion_matrix.png')
    plt.savefig(cm_fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Confusion Matrix Figure to: {cm_fig_path}")

    print("\n--- EXTRACTING QUALITATIVE ERRORS ---")
    
    # CHANGED: Isolate where Base DistilBERT failed but DAPT-DistilBERT succeeded
    error_df = df[(df['label'] != df['Pred_Base']) & (df['label'] == df['Pred_DAPT'])].copy()
    
    error_df['True_Sentiment'] = error_df['label'].map(LABEL_MAP)
    error_df['Base_Model_Guessed'] = error_df['Pred_Base'].map(LABEL_MAP)
    
    export_df = error_df[['text', 'True_Sentiment', 'Base_Model_Guessed']]
    
    errors_path = os.path.join(REPORTS_DIR, 'qualitative_errors_for_thesis.csv')
    export_df.to_csv(errors_path, index=False)
    print(f"Found {len(export_df)} specific sentences where Base DistilBERT failed but DAPT succeeded.")
    print(f"Saved qualitative errors to: {errors_path}")
    print("\nDone! Open 'qualitative_errors_for_thesis.csv' to pick real examples for Chapter 4.4.")

if __name__ == "__main__":
    main()