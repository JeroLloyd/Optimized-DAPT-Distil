"""
Comparative Error Analysis and Visualization Script.

This script evaluates and compares two sequence classification models. 
It performs batch inference using a baseline model and a domain-adapted model. 
The script calculates a confusion matrix, plots the results, and isolates 
specific text samples where domain adaptation corrected baseline classification errors.
"""

import os
import sys
import gc

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    DataCollatorWithPadding
)

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

# Adjust base directory if running directly from the source folder
if os.path.basename(SCRIPT_DIR) == "src":
    BASE_DIR = os.path.dirname(SCRIPT_DIR)
else:
    BASE_DIR = SCRIPT_DIR

DATA_PATH = os.path.join(BASE_DIR, 'data', '03_processed', 'FiReCS_Final', 'test.csv')
MODEL_A_DIR = os.path.join(BASE_DIR, 'models', 'model_a_base')
MODEL_B_DIR = os.path.join(BASE_DIR, 'models', 'model_b_dapt')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports', 'metrics')
FIGURES_DIR = os.path.join(BASE_DIR, 'reports', 'figures')

os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

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


def normalize_columns(df):
    """
    Standardizes dataset column names and removes rows with missing values.

    Args:
        df (pd.DataFrame): The raw test dataframe.

    Returns:
        pd.DataFrame: A cleaned dataframe containing only valid 'text' and 'label' columns.
    """
    if 'text' not in df.columns and 'review' in df.columns:
        df = df.rename(columns={'review': 'text'})
    if 'label' not in df.columns and 'sentiment' in df.columns:
        df = df.rename(columns={'sentiment': 'label'})
        
    df = df.dropna(subset=['text', 'label'])
    df['label'] = df['label'].astype(int)
    return df


def get_predictions_batched(model_path, texts, tokenizer_name):
    """
    Executes batch inference across a dataset using the Hugging Face Trainer.

    Args:
        model_path (str): The directory containing the model weights.
        texts (list): A list of text strings for inference.
        tokenizer_name (str): The identifier for the required tokenizer.

    Returns:
        np.ndarray: An array of predicted class integers.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    ds = Dataset.from_pandas(pd.DataFrame({"text": texts}))
    
    # Apply tokenization
    tokenized_ds = ds.map(
        lambda x: tokenizer(x["text"], truncation=True, max_length=128), 
        batched=True
    )
    
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()

    trainer = Trainer(
        model=model, 
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
    )
    
    print(f"-> Generating inferences for {os.path.basename(model_path)}...")
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
    Executes the comparative error analysis workflow.
    
    This function loads test data, extracts model predictions, computes the 
    confusion matrix, plots the results, and saves qualitative success cases 
    to a CSV file.
    """
    print(f"Project Root Detected: {BASE_DIR}")
    print("--- GENERATING QUALITATIVE ERROR DATA (A vs B) ---")

    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] Test data not found: {DATA_PATH}")
        return
    
    # --------------------------------------------------------------------------
    # 1. LOAD AND PREPARE DATA
    # --------------------------------------------------------------------------
    df_raw = pd.read_csv(DATA_PATH)
    df = normalize_columns(df_raw)
    
    texts = df['text'].tolist()
    true_labels = df['label'].tolist()

    # --------------------------------------------------------------------------
    # 2. RUN SYNCHRONIZED BATCH INFERENCES
    # --------------------------------------------------------------------------
    preds_a = get_predictions_batched(MODEL_A_DIR, texts, "distilbert-base-multilingual-cased")
    preds_b = get_predictions_batched(MODEL_B_DIR, texts, "distilbert-base-multilingual-cased")

    df['Pred_Base'] = preds_a
    df['Pred_DAPT'] = preds_b

    # --------------------------------------------------------------------------
    # 3. CONFUSION MATRIX GENERATION
    # --------------------------------------------------------------------------
    print("\n--- CONFUSION MATRIX (Model B - DAPT-DistilmBERT) ---")
    cm = confusion_matrix(true_labels, preds_b, labels=[0, 1, 2])
    
    cm_df = pd.DataFrame(
        cm, 
        index=["True Negative", "True Neutral", "True Positive"], 
        columns=["Pred Negative", "Pred Neutral", "Pred Positive"]
    )
    cm_df.to_csv(os.path.join(REPORTS_DIR, 'real_confusion_matrix.csv'))

    # Generate visual report
    plt.figure(figsize=(8, 6))
    sns.set_theme(style="white")
    ax = sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=["Negative", "Neutral", "Positive"],
        yticklabels=["Negative", "Neutral", "Positive"],
        annot_kws={"size": 14, "weight": "bold"},
        linewidths=1, linecolor='black'
    )
    
    plt.title("Confusion Matrix: DAPT-DistilmBERT (Model B)", fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Sentiment', fontweight='bold')
    plt.xlabel('Predicted Sentiment', fontweight='bold')
    plt.tight_layout()
    
    plt.savefig(os.path.join(FIGURES_DIR, '7_confusion_matrix.png'), dpi=300)
    plt.close()

    # --------------------------------------------------------------------------
    # 4. ISOLATE QUALITATIVE SUCCESS CASES
    # --------------------------------------------------------------------------
    # Filter for cases where the generic model failed and the adapted model succeeded
    success_cases = df[(df['label'] != df['Pred_Base']) & (df['label'] == df['Pred_DAPT'])].copy()
    
    success_cases['True_Sentiment'] = success_cases['label'].map(LABEL_MAP)
    success_cases['Base_Model_Error'] = success_cases['Pred_Base'].map(LABEL_MAP)
    
    export_df = success_cases[['text', 'True_Sentiment', 'Base_Model_Error']]
    export_df.to_csv(os.path.join(REPORTS_DIR, 'qualitative_errors_MODEL_A_vs_B.csv'), index=False)
    
    print(f"SUCCESS: Found {len(export_df)} cases where DAPT improved classification.")


if __name__ == "__main__":
    main()