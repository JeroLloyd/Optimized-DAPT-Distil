import os
import pandas as pd
import numpy as np
import torch
import shutil
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    set_seed
)
from sklearn.metrics import accuracy_score, f1_score

# --- PATH CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(BASE_DIR, 'data', '03_processed', 'FiReCS_Final')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Global Defaults
MAX_LEN = 128
set_seed(42)

# --- CONFIGURATION MATRIX (CRITICAL FOR HIERARCHY) ---
TRAINING_CONFIGS = [
    {
        # Model A: High LR to learn from scratch
        "name": "Model A (Base DistilBERT)",
        "base_model": "distilbert-base-multilingual-cased",
        "output_dir": os.path.join(MODELS_DIR, "model_a_base"),
        "learning_rate": 5e-5, 
        "num_epochs": 5,
        "batch_size": 32
    },
    {
        # Model B: Low LR to PRESERVE DAPT knowledge
        "name": "Model B (DAPT-DistilBERT)",
        "base_model": os.path.join(MODELS_DIR, "stage1_dapt_distilbert"),
        "output_dir": os.path.join(MODELS_DIR, "model_b_dapt"),
        "learning_rate": 2e-5,  # <-- THIS IS THE KEY
        "num_epochs": 8,        # <-- More epochs to settle
        "batch_size": 32
    },
    {
        # Model C: More epochs for the massive model
        "name": "Model C (XLM-R Base)",
        "base_model": "xlm-roberta-base",
        "output_dir": os.path.join(MODELS_DIR, "model_c_xlmr"),
        "learning_rate": 2e-5,
        "num_epochs": 10,
        "batch_size": 16
    }
]

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='macro')
    return {"accuracy": acc, "macro_f1": f1}

def train_model(config, tokenized_datasets, tokenizer):
    print(f"\n=== Training {config['name']} ===")
    
    if os.path.exists(config['output_dir']):
        shutil.rmtree(config['output_dir'])

    model = AutoModelForSequenceClassification.from_pretrained(
        config['base_model'], 
        num_labels=3
    )
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    training_args = TrainingArguments(
        output_dir=config['output_dir'],
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=config['learning_rate'],
        per_device_train_batch_size=config['batch_size'],
        per_device_eval_batch_size=config['batch_size'],
        num_train_epochs=config['num_epochs'],
        weight_decay=0.01,
        warmup_ratio=0.1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        report_to="none"
    )

    import time

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=4)]
    )

    # Reset GPU memory tracker before training starts
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        
    start_time = time.time()
    
    # Execute training
    trainer.train()
    
    # Calculate total duration and peak memory
    total_time = time.time() - start_time
    avg_epoch_time = total_time / config['num_epochs']
    
    print(f"\n--- HARDWARE METRICS: {config['name']} ---")
    print(f"Total Training Time: {total_time:.2f} seconds")
    print(f"Average Time per Epoch: {avg_epoch_time:.2f} seconds")
    
    if torch.cuda.is_available():
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        print(f"Peak GPU VRAM Allocated: {peak_vram_mb:.2f} MB")
    print("-------------------------------------------\n")

    trainer.save_model(config['output_dir'])

def normalize_columns(df):
    """Normalize column names to standard 'text' and 'label'."""
    df.columns = [c.lower().strip() for c in df.columns]
    
    # 1. Normalize Text Column
    text_candidates = ['review', 'text', 'content', 'sentence', 'body']
    for candidate in text_candidates:
        if candidate in df.columns:
            df = df.rename(columns={candidate: 'text'})
            break
    
    if 'text' not in df.columns:
        raise ValueError(f"Could not find text column. Available: {df.columns}")

    # 2. Normalize Label/Sentiment Column
    label_candidates = ['sentiment', 'label', 'target', 'class']
    label_col = None
    for candidate in label_candidates:
        if candidate in df.columns:
            label_col = candidate
            break
    
    if not label_col:
        raise ValueError(f"Could not find label column. Available: {df.columns}")

    # 3. Map Labels if they are strings
    first_val = df[label_col].iloc[0]
    if isinstance(first_val, str):
        label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
        # Handle potential capitalization
        df['label'] = df[label_col].astype(str).str.lower().map(label_map)
        
        # Check for unmapped values
        if df['label'].isnull().any():
            print(f"[WARNING] Some labels could not be mapped. Unique values in '{label_col}': {df[label_col].unique()}")
            df = df.dropna(subset=['label']) # Safe drop
            df['label'] = df['label'].astype(int)
    else:
        # Assume already integer
        df['label'] = df[label_col].astype(int)
    
    return df[['text', 'label']]

def main():
    print(f"Project Root: {BASE_DIR}")
    train_path = os.path.join(DATA_DIR, 'train.csv')
    val_path = os.path.join(DATA_DIR, 'val.csv')
    
    if not os.path.exists(train_path):
        print(f"[ERROR] Data missing: {train_path}")
        return

    print("Loading datasets...")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
    # --- ROBUST NORMALIZATION ---
    print("Normalizing columns...")
    try:
        train_df = normalize_columns(train_df)
        val_df = normalize_columns(val_df)
    except ValueError as e:
        print(f"[ERROR] Data format issue: {e}")
        return

    print(f"Training on {len(train_df)} samples, Validating on {len(val_df)} samples.")
    
    raw_datasets = DatasetDict({
        "train": Dataset.from_pandas(train_df),
        "validation": Dataset.from_pandas(val_df)
    })

    for config in TRAINING_CONFIGS:
        if "stage1_dapt" in config['base_model']:
            tokenizer_source = config['base_model']
        elif "xlm" in config['base_model']:
            tokenizer_source = "xlm-roberta-base"
        else:
            tokenizer_source = "distilbert-base-multilingual-cased"
            
        # Ensure tokenizer source exists
        if not os.path.exists(tokenizer_source) and '/' in tokenizer_source and not tokenizer_source.count('/') == 1: 
             # It's a path that doesn't exist, revert to base to avoid crash
             print(f"[WARNING] Tokenizer path {tokenizer_source} not found. Fallback to base.")
             tokenizer_source = "distilbert-base-multilingual-cased"

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)

        def tokenize_function(examples):
            return tokenizer(examples["text"], truncation=True, max_length=MAX_LEN)

        tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
        train_model(config, tokenized_datasets, tokenizer)

if __name__ == "__main__":
    main()