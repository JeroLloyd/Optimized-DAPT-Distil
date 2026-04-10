"""
Comparative Error Analysis Script.

This script isolates specific classification errors between two models. 
It loads the test dataset and runs batch inference using both a standard 
PyTorch model and a quantized ONNX model. The script then extracts cases 
where the baseline model predicts correctly and the quantized model fails. 
It exports these specific discrepancies to a CSV file for qualitative review.
"""

import os
import sys
import gc

import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    DataCollatorWithPadding
)
from optimum.onnxruntime import ORTModelForSequenceClassification

# ==============================================================================
# COMPATIBILITY PATCHES
# ==============================================================================
import transformers.utils
import transformers.modeling_utils
import transformers.models.auto

# Disable offline mode checks to prevent execution blocks
if not hasattr(transformers.utils, 'is_offline_mode'):
    transformers.utils.is_offline_mode = lambda: False

# ==============================================================================
# PATH CONFIGURATION
# ==============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(SCRIPT_DIR)
BASE_DIR = os.path.dirname(SRC_DIR)

DATA_PATH = os.path.join(BASE_DIR, 'data', '03_processed', 'FiReCS_Final', 'test.csv')
MODEL_C_DIR = os.path.join(BASE_DIR, 'models', 'model_c_xlmr')
MODEL_D_DIR = os.path.join(BASE_DIR, 'models', 'model_d_onnx')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports', 'metrics')

os.makedirs(REPORTS_DIR, exist_ok=True)

# Define integer-to-string mapping for human-readable output
LABEL_MAP = {0: "Negative", 1: "Neutral", 2: "Positive"}


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def clear_memory():
    """
    Forces garbage collection and clears the CUDA memory cache.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def safe_load_tokenizer(model_path, fallback_name):
    """
    Loads a tokenizer from a designated path. It falls back to a base model 
    if the local path fails.

    Args:
        model_path (str): The primary directory path containing the tokenizer.
        fallback_name (str): The Hugging Face identifier for the fallback model.

    Returns:
        PreTrainedTokenizer: The loaded tokenizer instance.
    """
    try:
        return AutoTokenizer.from_pretrained(model_path)
    except Exception:
        return AutoTokenizer.from_pretrained(fallback_name)


def get_predictions_batched(model_path, texts, tokenizer_name, is_onnx=False):
    """
    Executes batch inference across a dataset. It processes inputs differently 
    depending on the target runtime environment.

    Args:
        model_path (str): The directory containing the model weights.
        texts (list): A list of text strings for inference.
        tokenizer_name (str): The identifier for the required tokenizer.
        is_onnx (bool): A flag indicating if the model requires the ONNX runtime.

    Returns:
        np.ndarray: An array of predicted class integers.
    """
    tokenizer = safe_load_tokenizer(model_path, tokenizer_name)
    
    # Initialize the dataset and apply tokenization
    ds = Dataset.from_pandas(pd.DataFrame({"text": texts}))
    tokenized_ds = ds.map(
        lambda x: tokenizer(x["text"], truncation=True, max_length=128), 
        batched=True
    )
    
    print(f"-> Extracting inferences from {os.path.basename(model_path)}...")

    if is_onnx:
        # ----------------------------------------------------------------------
        # NATIVE ONNX INFERENCE LOOP
        # ----------------------------------------------------------------------
        model = ORTModelForSequenceClassification.from_pretrained(
            model_path, 
            provider="CPUExecutionProvider",
            use_io_binding=False 
        )
        
        # Prepare data loader for ONNX processing
        tokenized_ds = tokenized_ds.remove_columns(["text"])
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
        dataloader = DataLoader(tokenized_ds, batch_size=32, collate_fn=data_collator)
        
        all_preds = []
        for batch in dataloader:
            # Remove incompatible token type IDs for specific models
            if "distilbert" in tokenizer_name.lower():
                batch.pop("token_type_ids", None)
                
            outputs = model(**batch)
            logits = outputs.logits
            
            if torch.is_tensor(logits):
                logits = logits.detach().cpu().numpy()
                
            batch_preds = np.argmax(logits, axis=-1)
            all_preds.extend(batch_preds)
            
        y_pred = np.array(all_preds)
        del model
        
    else:
        # ----------------------------------------------------------------------
        # STANDARD PYTORCH TRAINER INFERENCE
        # ----------------------------------------------------------------------
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.eval()

        trainer = Trainer(
            model=model, 
            tokenizer=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
        )
        
        output = trainer.predict(tokenized_ds)
        y_pred = np.argmax(output.predictions, axis=-1)
        
        del model, trainer

    clear_memory()
    return y_pred


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def main():
    """
    Executes the qualitative error extraction workflow.
    
    This function loads the target data and runs synchronous batch inference.
    It isolates the cases where the baseline model provides the correct answer 
    while the quantized model fails. It then exports these specific cases to a CSV.
    """
    print(f"Project Root Detected: {BASE_DIR}")
    print("--- GENERATING QUALITATIVE ERROR DATA (C vs D) ---")

    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] Test data not found: {DATA_PATH}")
        return
    
    # Load dataset and standardize the text column name
    df = pd.read_csv(DATA_PATH)
    text_col = 'review' if 'review' in df.columns else 'text'
    texts = df[text_col].tolist()

    # --------------------------------------------------------------------------
    # PHASE 1: Run Synchronized Batch Inferences
    # --------------------------------------------------------------------------
    preds_c = get_predictions_batched(MODEL_C_DIR, texts, "xlm-roberta-base", is_onnx=False)
    preds_d = get_predictions_batched(MODEL_D_DIR, texts, "distilbert-base-multilingual-cased", is_onnx=True)

    df['Pred_XLM_R'] = preds_c
    df['Pred_Optimized_DAPT'] = preds_d

    # --------------------------------------------------------------------------
    # PHASE 2: Isolate Competitive Edge Errors
    # --------------------------------------------------------------------------
    # Filter for rows where Model C is correct and Model D is incorrect
    edge_errors = df[(df['label'] != df['Pred_Optimized_DAPT']) & (df['label'] == df['Pred_XLM_R'])].copy()
    
    # Map integer labels to readable string categories
    edge_errors['True_Sentiment'] = edge_errors['label'].map(LABEL_MAP)
    edge_errors['XLM_R_Correct_Guess'] = edge_errors['Pred_XLM_R'].map(LABEL_MAP)
    edge_errors['Model_D_Error_Guess'] = edge_errors['Pred_Optimized_DAPT'].map(LABEL_MAP)
    
    export_df = edge_errors[[text_col, 'True_Sentiment', 'Model_D_Error_Guess']]
    
    # Export findings to a CSV file
    out_path = os.path.join(REPORTS_DIR, 'qualitative_errors_Model_C_vs_D.csv')
    export_df.to_csv(out_path, index=False)
    
    print("\n--- ANALYSIS SUMMARY ---")
    print(f"Found {len(export_df)} cases where XLM-R outperformed Optimized DAPT.")
    print(f"Results saved to: {out_path}")
    print("Action: Review this CSV to identify linguistic patterns that quantization might affect.")


if __name__ == "__main__":
    main()