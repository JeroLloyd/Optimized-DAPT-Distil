# FILE: 06_train_stage2_finetune.py
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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=4)]
    )

    trainer.train()
    trainer.save_model(config['output_dir'])
    tokenizer.save_pretrained(config['output_dir'])

def main():
    print(f"Project Root: {BASE_DIR}")
    train_path = os.path.join(DATA_DIR, 'train.csv')
    val_path = os.path.join(DATA_DIR, 'val.csv')
    
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
    if 'review' in train_df.columns:
        train_df = train_df.rename(columns={'review': 'text'})
        val_df = val_df.rename(columns={'review': 'text'})
    
    label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
    train_df['label'] = train_df['sentiment'].map(label_map)
    val_df['label'] = val_df['sentiment'].map(label_map)
    
    raw_datasets = DatasetDict({
        "train": Dataset.from_pandas(train_df[['text', 'label']]),
        "validation": Dataset.from_pandas(val_df[['text', 'label']])
    })

    for config in TRAINING_CONFIGS:
        if "stage1_dapt" in config['base_model']:
            tokenizer_source = config['base_model']
        elif "xlm" in config['base_model']:
            tokenizer_source = "xlm-roberta-base"
        else:
            tokenizer_source = "distilbert-base-multilingual-cased"
            
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)

        def tokenize_function(examples):
            return tokenizer(examples["text"], truncation=True, max_length=MAX_LEN)

        tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
        train_model(config, tokenized_datasets, tokenizer)

if __name__ == "__main__":
    main()