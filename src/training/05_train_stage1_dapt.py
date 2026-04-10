"""
Stage 1 Domain-Adaptive Pre-Training Execution Script.

This module handles the first stage of training for a language model.
It applies domain-adaptive pre-training to a baseline checkpoint using a custom corpus.
The script configures deterministic seeding, initializes the model and tokenizer,
executes the training loop, and exports performance metrics and loss curves.
"""

import os
import shutil
import time
import random

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    set_seed
)

# ==============================================================================
# DETERMINISTIC SEED LOCK
# ==============================================================================
def lock_environmental_seeds(seed=42):
    """
    Secures all random number generators to ensure reproducible training runs.

    Args:
        seed (int): The numerical seed value to apply across all libraries.
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

lock_environmental_seeds(42)

# ==============================================================================
# PATH CONFIGURATION
# ==============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(SCRIPT_DIR)
BASE_DIR = os.path.dirname(SRC_DIR)

PROCESSED_DIR = os.path.join(BASE_DIR, 'data', '03_processed')
MODEL_OUTPUT_DIR = os.path.join(BASE_DIR, 'models', 'stage1_dapt_distilmbert')
TRAIN_FILE = os.path.join(PROCESSED_DIR, 'hybrid_corpus.txt')
DAPT_LOSS_DIR = os.path.join(BASE_DIR, 'reports', 'figures', '09_dapt_loss')

os.makedirs(DAPT_LOSS_DIR, exist_ok=True)

# ==============================================================================
# OPTIMIZED HYPERPARAMETERS FOR DEEP ADAPTATION
# ==============================================================================
MODEL_CHECKPOINT = "distilbert-base-multilingual-cased"
BATCH_SIZE = 16
GRADIENT_ACCUMULATION = 2  
LEARNING_RATE = 3e-5       
EPOCHS = 10                
MLM_PROBABILITY = 0.15


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def main():
    """
    Executes the primary training pipeline.

    This function manages data loading, tokenization, model configuration,
    the training loop, and metric logging. It exports the finalized model,
    performance summaries, and loss visualizations.
    """
    print(f"Project Root Detected: {BASE_DIR}")
    print(f"--- STAGE 1: HIGH-CAPACITY DAPT (LR: {LEARNING_RATE}, Epochs: {EPOCHS}) ---")
    
    # Validate training file availability
    if not os.path.exists(TRAIN_FILE):
        print(f"[ERROR] Training data not found at: {TRAIN_FILE}")
        return

    # Reset output directory to prevent state conflicts
    if os.path.exists(MODEL_OUTPUT_DIR):
        print(f"Clearing old model directory: {MODEL_OUTPUT_DIR}")
        shutil.rmtree(MODEL_OUTPUT_DIR)

    # Initialize tokenizer and dataset
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    datasets = load_dataset('text', data_files={'train': TRAIN_FILE})
    
    def tokenize_function(examples):
        """
        Applies the tokenizer to a batch of text inputs.
        """
        return tokenizer(examples["text"], truncation=True, max_length=128)
    
    # Apply tokenization across the dataset using multiprocessing
    tokenized_datasets = datasets.map(
        tokenize_function, batched=True, num_proc=4, remove_columns=["text"]
    )
    
    # Prepare the data collator for masked language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=MLM_PROBABILITY
    )
    
    # Initialize the base model
    model = AutoModelForMaskedLM.from_pretrained(MODEL_CHECKPOINT)
    
    # Configure training parameters
    training_args = TrainingArguments(
        output_dir=MODEL_OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        save_strategy="no", 
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,     
        warmup_ratio=0.15,     
        logging_strategy="steps",
        logging_steps=50, 
        fp16=torch.cuda.is_available(),
        report_to="none"
    )
    
    # Construct the trainer object
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        data_collator=data_collator,
    )
    
    # Track memory usage if hardware supports it
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        
    start_time = time.time()
    
    # Execute the training loop
    trainer.train()
    
    total_time = time.time() - start_time
    
    # Calculate performance metrics
    stage1_dapt_avg_epoch_time = total_time / EPOCHS
    stage1_dapt_val_epochs = EPOCHS
    
    peak_vram_mb = 0
    if torch.cuda.is_available():
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    # Output performance summary
    print("\n--- DAPT PERFORMANCE SUMMARY ---")
    print(f"Total Training Time: {total_time:.2f} seconds")
    print(f"Average Epoch Time: {stage1_dapt_avg_epoch_time:.2f} seconds")
    print(f"Total Epochs Run: {stage1_dapt_val_epochs}")
    print(f"Peak GPU VRAM Allocated: {peak_vram_mb:.2f} MB")
        
    # Extract and save loss history
    log_history = trainer.state.log_history
    losses = [log for log in log_history if 'loss' in log]
    
    if losses:
        df_loss = pd.DataFrame(losses)
        csv_path_loss = os.path.join(DAPT_LOSS_DIR, 'dapt_loss.csv')
        df_loss.to_csv(csv_path_loss, index=False)
        print(f"SUCCESS: Loss history saved to {csv_path_loss}")
        
        # Export training summary data
        summary_data = [{
            "Model_Checkpoint": MODEL_CHECKPOINT,
            "Learning_Rate": LEARNING_RATE,
            "Batch_Size": BATCH_SIZE * GRADIENT_ACCUMULATION,
            "MLM_Probability": MLM_PROBABILITY,
            "Total_Epochs_Run": stage1_dapt_val_epochs,
            "Total_Training_Time_s": round(total_time, 2),
            "Avg_Epoch_Time_s": round(stage1_dapt_avg_epoch_time, 2),
            "Peak_VRAM_MB": round(peak_vram_mb, 2)
        }]
        
        df_summary = pd.DataFrame(summary_data)
        csv_path_summary = os.path.join(DAPT_LOSS_DIR, 'dapt_training_summary.csv')
        df_summary.to_csv(csv_path_summary, index=False)
        print(f"SUCCESS: Training summary saved to {csv_path_summary}")
        
        # Generate and save loss curve plot
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
    
    # Save finalized model and tokenizer
    trainer.save_model(MODEL_OUTPUT_DIR)
    tokenizer.save_pretrained(MODEL_OUTPUT_DIR)


if __name__ == "__main__":
    main()