import os
import pandas as pd
import numpy as np
import torch
import shutil
import gc
import random
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
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# --- DETERMINISTIC SEED LOCK ---
def lock_environmental_seeds(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

lock_environmental_seeds(42)

# --- PATH CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(SCRIPT_DIR)
BASE_DIR = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(BASE_DIR, 'data', '03_processed', 'FiReCS_Final')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports', 'metrics')
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# Global Defaults
MAX_LEN = 128
BASE_LIMIT_RATES = [1e-5, 2e-5]
SEARCH_RATES = [1e-5, 2e-5, 3e-5, 5e-5]

TRAINING_CONFIGS = [
    {
        "name": "Model A (Base DistilBERT)",
        "base_model": "distilbert-base-multilingual-cased",
        "output_dir": os.path.join(MODELS_DIR, "model_a_base"),
        "learning_rates": BASE_LIMIT_RATES, 
        "num_epochs": 3, 
        "batch_size": 32
    },
    {
        "name": "Model B (DAPT-DistilBERT)",
        "base_model": os.path.join(MODELS_DIR, "stage1_dapt_distilbert"),
        "output_dir": os.path.join(MODELS_DIR, "model_b_dapt"),
        "learning_rates": SEARCH_RATES, 
        "num_epochs": 8,
        "batch_size": 32
    },
    {
        "name": "Model C (XLM-R Base)",
        "base_model": "xlm-roberta-base",
        "output_dir": os.path.join(MODELS_DIR, "model_c_xlmr"),
        "learning_rates": SEARCH_RATES, 
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

def normalize_columns(df):
    """
    Cleans the dataframe and ensures correct data types for training.
    """
    if 'text' not in df.columns and 'review' in df.columns:
        df = df.rename(columns={'review': 'text'})
    if 'label' not in df.columns and 'sentiment' in df.columns:
        df = df.rename(columns={'sentiment': 'label'})
    
    # CRITICAL FIX: Cast labels to integer to avoid RuntimeError in PyTorch
    df = df.dropna(subset=['text', 'label'])
    df['label'] = df['label'].astype(int)
    
    return df[['text', 'label']]

def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def main():
    print("=== STAGE 2: SUPERVISED FINE-TUNING (GRID SEARCH + VRAM LOGGING) ===")
    
    train_path = os.path.join(DATA_DIR, 'train.csv')
    val_path = os.path.join(DATA_DIR, 'val.csv')

    if not os.path.exists(train_path) or not os.path.exists(val_path):
        print(f"[ERROR] Datasets missing in {DATA_DIR}")
        return

    train_df = normalize_columns(pd.read_csv(train_path))
    val_df = normalize_columns(pd.read_csv(val_path))
    
    raw_datasets = DatasetDict({
        "train": Dataset.from_pandas(train_df),
        "validation": Dataset.from_pandas(val_df)
    })

    all_search_results = []

    for config in TRAINING_CONFIGS:
        print(f"\n--- Initiating Grid Search for {config['name']} ---")
        
        tokenizer_source = config['base_model']
        if not os.path.exists(tokenizer_source) and '/' in tokenizer_source: 
             tokenizer_source = "distilbert-base-multilingual-cased"

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)

        def tokenize_function(examples):
            return tokenizer(examples["text"], truncation=True, padding='max_length', max_length=MAX_LEN)

        tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        
        best_f1 = 0.0
        best_lr = None
        best_temp_dir = ""

        for lr in config['learning_rates']:
            print(f"\n[TESTING] Learning Rate: {lr}")
            temp_output_dir = f"{config['output_dir']}_TEMP_LR_{lr}"
            
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            model = AutoModelForSequenceClassification.from_pretrained(
                config['base_model'], 
                num_labels=3,
                problem_type="single_label_classification"
            )

            training_args = TrainingArguments(
                output_dir=temp_output_dir,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                learning_rate=lr,
                per_device_train_batch_size=config['batch_size'],
                per_device_eval_batch_size=config['batch_size'],
                num_train_epochs=config['num_epochs'],
                weight_decay=0.01,
                warmup_ratio=0.1,
                load_best_model_at_end=True,
                metric_for_best_model="macro_f1",
                greater_is_better=True,
                save_total_limit=1,
                fp16=torch.cuda.is_available(),
                report_to="none"
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_datasets["train"],
                eval_dataset=tokenized_datasets["validation"],
                tokenizer=tokenizer,
                data_collator=data_collator,
                compute_metrics=compute_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=4)]
            )

            trainer.train()
            metrics = trainer.evaluate()
            current_f1 = metrics['eval_macro_f1']
            
            # --- CRITICAL ADDITION: Explicitly dump the model and config to the root directory ---
            trainer.save_model(temp_output_dir)
            tokenizer.save_pretrained(temp_output_dir)
            
            vram_mb = 0
            if torch.cuda.is_available():
                vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

            print(f"[RESULT] LR {lr} | F1: {current_f1:.4f} | Peak VRAM: {vram_mb:.2f} MB")
            
            all_search_results.append({
                "Model_Name": config['name'],
                "Learning_Rate": lr,
                "Macro_F1": current_f1,
                "Peak_VRAM_MB": round(vram_mb, 2)
            })

            if current_f1 > best_f1:
                best_f1 = current_f1
                best_lr = lr
                best_temp_dir = temp_output_dir
            
            del model, trainer
            clear_memory()

        print(f"\n[WINNER] Best LR for {config['name']} is {best_lr} (F1: {best_f1:.4f})")
        
        if os.path.exists(config['output_dir']):
            shutil.rmtree(config['output_dir'])
            
        shutil.copytree(best_temp_dir, config['output_dir'])
        
        for lr in config['learning_rates']:
            temp_dir = f"{config['output_dir']}_TEMP_LR_{lr}"
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    summary_df = pd.DataFrame(all_search_results)
    summary_path = os.path.join(REPORTS_DIR, "finetuning_grid_search_results.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n[SUCCESS] Grid search summary saved to {summary_path}")

if __name__ == "__main__":
    main()