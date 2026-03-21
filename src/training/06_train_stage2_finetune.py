# FILE: 06_train_stage2_finetune.py
import os
import time
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

STAGE2_LOGS_DIR = os.path.join(REPORTS_DIR, 'stage2_logs')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(STAGE2_LOGS_DIR, exist_ok=True)

# Global Defaults
MAX_LEN = 128
UNIVERSAL_SEARCH_RATES = [1e-5, 1.2e-5, 1.5e-5] 
MAX_EPOCHS = 15

# --- EQUITABLE CONFIGURATION (SCIENTIFICALLY SOUND) ---
TRAINING_CONFIGS = [
    {
        "name": "Model A (Base DistilmBERT)",
        "base_model": "distilbert-base-multilingual-cased",
        "output_dir": os.path.join(MODELS_DIR, "model_a_base"),
        "learning_rates": UNIVERSAL_SEARCH_RATES, 
        "num_epochs": MAX_EPOCHS, 
        "batch_size": 16, 
        "gradient_accumulation_steps": 1,
        "llrd_decay": 0.95  # SYNCED: Equal treatment for the baseline
    },
    {
        "name": "Model B (DAPT-DistilmBERT)",
        "base_model": os.path.join(MODELS_DIR, "stage1_dapt_distilmbert"),
        "output_dir": os.path.join(MODELS_DIR, "model_b_dapt"),
        "learning_rates": UNIVERSAL_SEARCH_RATES, 
        "num_epochs": MAX_EPOCHS,
        "batch_size": 16, 
        "gradient_accumulation_steps": 1,
        "llrd_decay": 0.95  # SYNCED: Equal treatment for the proposed model
    },
    {
        "name": "Model C (XLM-R Base)",
        "base_model": "xlm-roberta-base",
        "output_dir": os.path.join(MODELS_DIR, "model_c_xlmr"),
        "learning_rates": UNIVERSAL_SEARCH_RATES, 
        "num_epochs": MAX_EPOCHS,
        "batch_size": 16,
        "gradient_accumulation_steps": 2,
        "llrd_decay": 0.95
    }
]

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='macro')
    return {"accuracy": acc, "macro_f1": f1}

def normalize_columns(df):
    if 'text' not in df.columns and 'review' in df.columns:
        df = df.rename(columns={'review': 'text'})
    if 'label' not in df.columns and 'sentiment' in df.columns:
        df = df.rename(columns={'sentiment': 'label'})
    df = df.dropna(subset=['text', 'label'])
    df['label'] = df['label'].astype(int)
    return df[['text', 'label']]

def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# --- DYNAMIC LLRD LOGIC ---
def get_optimizer_grouped_parameters(model, base_lr, decay_factor, weight_decay=0.01):
    """Assigns dynamically passed decaying learning rates to model layers."""
    is_distilbert = hasattr(model, "distilbert")
    
    if is_distilbert:
        layers = model.distilbert.transformer.layer
        embeddings = model.distilbert.embeddings
    else:
        layers = model.roberta.encoder.layer
        embeddings = model.roberta.embeddings
        
    num_layers = len(layers)
    optimizer_grouped_parameters = []
    
    # 1. Classifier Head (Full LR)
    optimizer_grouped_parameters.append({
        "params": [p for n, p in model.named_parameters() if "classifier" in n or "pre_classifier" in n],
        "weight_decay": weight_decay,
        "lr": base_lr,
    })
    
    # 2. Transformer Layers
    for i in range(num_layers - 1, -1, -1):
        lr = base_lr * (decay_factor ** (num_layers - i))
        optimizer_grouped_parameters.append({
            "params": layers[i].parameters(),
            "weight_decay": weight_decay,
            "lr": lr,
        })
        
    # 3. Embeddings
    lr_embed = base_lr * (decay_factor ** (num_layers + 1))
    optimizer_grouped_parameters.append({
        "params": embeddings.parameters(),
        "weight_decay": weight_decay,
        "lr": lr_embed,
    })
    return optimizer_grouped_parameters

def main():
    print("=== STAGE 2: EQUITABLE GRID SEARCH (SCIENTIFICALLY ALIGNED) ===")
    
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

        tokenized_datasets = raw_datasets.map(
            lambda x: tokenizer(x["text"], truncation=True, padding='max_length', max_length=MAX_LEN), 
            batched=True
        )
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        
        best_f1 = 0.0
        best_lr, best_temp_dir = None, ""

        for lr in config['learning_rates']:
            print(f"\n[TESTING] Base LR: {lr} | Applying LLRD: {config['llrd_decay']}")
            temp_output_dir = f"{config['output_dir']}_TEMP_LR_{lr}"
            
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            model = AutoModelForSequenceClassification.from_pretrained(
                config['base_model'], num_labels=3, problem_type="single_label_classification"
            )

            # Apply dynamic decay factor (now equal for all)
            grouped_params = get_optimizer_grouped_parameters(model, lr, config['llrd_decay'], weight_decay=0.01)
            optimizer = torch.optim.AdamW(grouped_params)

            training_args = TrainingArguments(
                output_dir=temp_output_dir,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                learning_rate=lr,
                per_device_train_batch_size=config['batch_size'], 
                per_device_eval_batch_size=config['batch_size'],
                gradient_accumulation_steps=config['gradient_accumulation_steps'],
                num_train_epochs=config['num_epochs'],
                weight_decay=0.01,
                warmup_ratio=0.10,
                load_best_model_at_end=True,
                metric_for_best_model="macro_f1",
                greater_is_better=True,
                save_total_limit=1,
                fp16=torch.cuda.is_available(),
                report_to="none",
                label_smoothing_factor=0.1
            )

            trainer = Trainer(
                model=model, args=training_args, 
                train_dataset=tokenized_datasets["train"], eval_dataset=tokenized_datasets["validation"],
                tokenizer=tokenizer, data_collator=data_collator,
                compute_metrics=compute_metrics,
                optimizers=(optimizer, None),
                callbacks=[EarlyStoppingCallback(early_stopping_patience=2)] # Aligned with Table 7
            )

            # Run training and capture output metrics
            train_result = trainer.train()
            
            # Extract time and epoch data
            total_train_time = train_result.metrics.get("train_runtime", 0.0)
            epochs_run = trainer.state.epoch
            avg_epoch_time = total_train_time / epochs_run if epochs_run > 0 else 0.0
            
            vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0
            metrics = trainer.evaluate()
            current_f1 = metrics['eval_macro_f1']
            
            print(f"[RESULT] LR {lr} | F1: {current_f1:.4f} | Peak VRAM: {vram_mb:.2f} MB | Total Time: {total_train_time:.2f}s | Avg Epoch Time: {avg_epoch_time:.2f}s")
            
            log_history = trainer.state.log_history
            if log_history:
                df_log = pd.DataFrame(log_history)
                df_log['peak_vram_mb'] = round(vram_mb, 2)
                df_log['total_train_time_s'] = round(total_train_time, 2)
                df_log['avg_epoch_time_s'] = round(avg_epoch_time, 2)
                clean_name = config['name'].split("(")[0].strip().replace(" ", "_").lower()
                log_filename = f"{clean_name}_lr_{lr}.csv"
                log_filepath = os.path.join(STAGE2_LOGS_DIR, log_filename)
                df_log.to_csv(log_filepath, index=False)

            trainer.save_model(temp_output_dir)
            tokenizer.save_pretrained(temp_output_dir)
            
            all_search_results.append({
                "Model_Name": config['name'], 
                "Learning_Rate": lr, 
                "Macro_F1": current_f1, 
                "Peak_VRAM_MB": round(vram_mb, 2),
                "Total_Time_s": round(total_train_time, 2),
                "Avg_Epoch_Time_s": round(avg_epoch_time, 2)
            })

            if current_f1 > best_f1:
                best_f1, best_lr, best_temp_dir = current_f1, lr, temp_output_dir
            
            del model, trainer
            clear_memory()

        print(f"\n[WINNER] {config['name']} Best LR: {best_lr} (F1: {best_f1:.4f})")
        
        if os.path.exists(config['output_dir']): shutil.rmtree(config['output_dir'])
        shutil.copytree(best_temp_dir, config['output_dir'])
        
        for lr in config['learning_rates']:
            td = f"{config['output_dir']}_TEMP_LR_{lr}"
            if os.path.exists(td): shutil.rmtree(td)

    summary_df = pd.DataFrame(all_search_results)
    summary_path = os.path.join(REPORTS_DIR, "finetuning_grid_search_results.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n[SUCCESS] Stage 2 Complete. Champion models and logs saved.")

if __name__ == "__main__":
    main()