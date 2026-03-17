# FILE: 05_train_stage1_dapt.py
import os
import shutil
import time
import torch
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    set_seed
)
from datasets import load_dataset

# --- DETERMINISTIC SEED LOCK ---
def lock_environmental_seeds(seed=42):
    """
    Locks all random seeds and forces deterministic algorithms 
    to ensure the DAPT phase produces identical weights on every run.
    """
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

# Execute the lock immediately
lock_environmental_seeds(42)

# --- PATH CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(SCRIPT_DIR)
BASE_DIR = os.path.dirname(SRC_DIR)
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', '03_processed')
MODEL_OUTPUT_DIR = os.path.join(BASE_DIR, 'models', 'stage1_dapt_distilbert')
TRAIN_FILE = os.path.join(PROCESSED_DIR, 'hybrid_corpus.txt')

# Create specific report directory
DAPT_LOSS_DIR = os.path.join(BASE_DIR, 'reports', 'figures', '09_dapt_loss')
os.makedirs(DAPT_LOSS_DIR, exist_ok=True)

# --- OPTIMIZED HYPERPARAMETERS ---
MODEL_CHECKPOINT = "distilbert-base-multilingual-cased"
BATCH_SIZE = 16

# CRITICAL SETTINGS FOR STABILITY
LEARNING_RATE = 2e-5  
EPOCHS = 8  
MLM_PROBABILITY = 0.15

def main():
    print(f"Project Root Detected: {BASE_DIR}")
    print(f"--- STAGE 1: DETERMINISTIC DOMAIN-ADAPTIVE PRE-TRAINING ({EPOCHS} EPOCHS) ---")
    
    if not os.path.exists(TRAIN_FILE):
        print(f"[ERROR] Training data not found at: {TRAIN_FILE}")
        return

    if os.path.exists(MODEL_OUTPUT_DIR):
        print(f"Clearing old model directory: {MODEL_OUTPUT_DIR}")
        shutil.rmtree(MODEL_OUTPUT_DIR)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    datasets = load_dataset('text', data_files={'train': TRAIN_FILE})
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=128)
    
    tokenized_datasets = datasets.map(
        tokenize_function, batched=True, num_proc=4, remove_columns=["text"]
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=MLM_PROBABILITY
    )
    
    model = AutoModelForMaskedLM.from_pretrained(MODEL_CHECKPOINT)
    
    training_args = TrainingArguments(
        output_dir=MODEL_OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        save_strategy="no", 
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_ratio=0.06,
        logging_strategy="steps",
        logging_steps=100, 
        fp16=torch.cuda.is_available(),
        report_to="none"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        data_collator=data_collator,
    )
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        
    start_time = time.time()
    
    # Execute Training
    trainer.train()
    
    total_time = time.time() - start_time
    
    # Capture Peak VRAM
    peak_vram_mb = 0
    if torch.cuda.is_available():
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    print(f"\n--- DAPT PERFORMANCE SUMMARY ---")
    print(f"Total Training Time: {total_time:.2f} seconds")
    print(f"Peak GPU VRAM Allocated: {peak_vram_mb:.2f} MB")
        
    # Extract Loss History
    log_history = trainer.state.log_history
    losses = [log for log in log_history if 'loss' in log]
    
    if losses:
        # Save Loss Data AND VRAM to existing CSV
        df_loss = pd.DataFrame(losses)
        df_loss['peak_vram_mb'] = round(peak_vram_mb, 2)
        
        csv_path = os.path.join(DAPT_LOSS_DIR, 'dapt_loss.csv')
        df_loss.to_csv(csv_path, index=False)
        print(f"SUCCESS: Training metrics (Loss + VRAM) saved to {csv_path}")
        
        # Plot Loss Curve
        plt.figure(figsize=(8, 5))
        plt.plot(df_loss['step'], df_loss['loss'], label='Training Loss (MLM)', color='#003366', linewidth=2)
        plt.xlabel('Training Steps', fontweight='bold')
        plt.ylabel('Cross-Entropy Loss', fontweight='bold')
        plt.title('Domain-Adaptive Pre-Training Loss Reduction', fontweight='bold', pad=15)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        
        plot_path = os.path.join(DAPT_LOSS_DIR, 'dapt_loss_curve.png')
        plt.savefig(plot_path, dpi=300)
        plt.close()

    print("------------------------------------\n")
    
    trainer.save_model(MODEL_OUTPUT_DIR)
    tokenizer.save_pretrained(MODEL_OUTPUT_DIR)

if __name__ == "__main__":
    main()