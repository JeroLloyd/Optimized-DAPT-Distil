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
from optimum.onnxruntime import ORTModelForSequenceClassification

# --- PATH CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Resolve root directory whether script is in root or src/
if os.path.basename(SCRIPT_DIR) == "src":
    BASE_DIR = os.path.dirname(SCRIPT_DIR)
else:
    BASE_DIR = SCRIPT_DIR

DATA_PATH = os.path.join(BASE_DIR, 'data', '03_processed', 'FiReCS_Final', 'test.csv')
MODEL_C_DIR = os.path.join(BASE_DIR, 'models', 'model_c_xlmr')
MODEL_D_DIR = os.path.join(BASE_DIR, 'models', 'model_d_onnx')
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

    print("Loading Model C (XLM-R)...")
    tokenizer_c = safe_load_tokenizer(MODEL_C_DIR, "xlm-roberta-base")
    model_c = AutoModelForSequenceClassification.from_pretrained(MODEL_C_DIR)
    model_c.eval()

    print("Loading Model D (Optimized DAPT)...")
    tokenizer_d = safe_load_tokenizer(MODEL_D_DIR, "distilbert-base-multilingual-cased")
    model_d = ORTModelForSequenceClassification.from_pretrained(MODEL_D_DIR, provider="CPUExecutionProvider")

    def get_predictions(tokenizer, model, is_onnx=False):
        preds = []
        print(f"Predicting {len(texts)} sequences...")
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
            # Remove token_type_ids for distilbert/ONNX compatibility if present
            if is_onnx and "token_type_ids" in inputs:
                inputs.pop("token_type_ids")

            if is_onnx:
                outputs = model(**inputs)
                logits = outputs.logits
            else:
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits
                    
            pred = np.argmax(logits.numpy() if is_onnx else logits.cpu().numpy(), axis=1)[0]
            preds.append(pred)
        return preds

    print("\nRunning inferences for XLM-R...")
    preds_c = get_predictions(tokenizer_c, model_c, is_onnx=False)
    
    print("Running inferences for Optimized DAPT...")
    preds_d = get_predictions(tokenizer_d, model_d, is_onnx=True)

    df['Pred_XLM_R'] = preds_c
    df['Pred_Optimized_DAPT'] = preds_d

    print("\n--- CONFUSION MATRIX (Model D) ---")
    cm = confusion_matrix(true_labels, preds_d, labels=[0, 1, 2])
    
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
    
    plt.title("Confusion Matrix: Optimized DAPT (Model D)", fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Sentiment', fontsize=13, fontweight='bold')
    plt.xlabel('Predicted Sentiment', fontsize=13, fontweight='bold')
    plt.tight_layout()

    cm_fig_path = os.path.join(FIGURES_DIR, '7_confusion_matrix.png')
    plt.savefig(cm_fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Confusion Matrix Figure to: {cm_fig_path}")

    print("\n--- EXTRACTING QUALITATIVE ERRORS ---")
    
    error_df = df[(df['label'] != df['Pred_Optimized_DAPT']) & (df['label'] == df['Pred_XLM_R'])].copy()
    
    error_df['True_Sentiment'] = error_df['label'].map(LABEL_MAP)
    error_df['Model_D_Guessed'] = error_df['Pred_Optimized_DAPT'].map(LABEL_MAP)
    
    export_df = error_df[['text', 'True_Sentiment', 'Model_D_Guessed']]
    
    errors_path = os.path.join(REPORTS_DIR, 'qualitative_errors_for_thesis.csv')
    export_df.to_csv(errors_path, index=False)
    print(f"Found {len(export_df)} specific sentences where XLM-R beat the Optimized model.")
    print(f"Saved qualitative errors to: {errors_path}")
    print("\nDone! Open 'qualitative_errors_for_thesis.csv' to pick real examples for Chapter 4.4.")

if __name__ == "__main__":
    main()