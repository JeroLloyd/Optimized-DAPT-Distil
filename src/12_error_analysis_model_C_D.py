import os
import sys
import pandas as pd
import numpy as np
import torch

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

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from optimum.onnxruntime import ORTModelForSequenceClassification

# --- PATH CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = SCRIPT_DIR if os.path.basename(SCRIPT_DIR) != "src" else os.path.dirname(SCRIPT_DIR)

# Resolve root directory whether script is in root or src/
if os.path.basename(SCRIPT_DIR) == "src":
    BASE_DIR = os.path.dirname(SCRIPT_DIR)
else:
    BASE_DIR = SCRIPT_DIR

DATA_PATH = os.path.join(BASE_DIR, 'data', '03_processed', 'FiReCS_Final', 'test.csv')
MODEL_C_DIR = os.path.join(BASE_DIR, 'models', 'model_c_xlmr')
MODEL_D_DIR = os.path.join(BASE_DIR, 'models', 'model_d_onnx')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports', 'metrics')

os.makedirs(REPORTS_DIR, exist_ok=True)

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

    print("\n--- EXTRACTING QUALITATIVE ERRORS ---")
    
    error_df = df[(df['label'] != df['Pred_Optimized_DAPT']) & (df['label'] == df['Pred_XLM_R'])].copy()
    
    error_df['True_Sentiment'] = error_df['label'].map(LABEL_MAP)
    error_df['Model_D_Guessed'] = error_df['Pred_Optimized_DAPT'].map(LABEL_MAP)
    
    export_df = error_df[['text', 'True_Sentiment', 'Model_D_Guessed']]
    
    errors_path = os.path.join(REPORTS_DIR, 'qualitative_errors_for_thesis_Model_C_D.csv')
    export_df.to_csv(errors_path, index=False)
    print(f"Found {len(export_df)} specific sentences where XLM-R beat the Optimized model.")
    print(f"Saved qualitative errors to: {errors_path}")
    print("\nDone! Open 'qualitative_errors_for_thesis_Model_C_D.csv' to pick real examples for Chapter 4.4.")

if __name__ == "__main__":
    main()