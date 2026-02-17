# FILE: 05_train_stage1_dapt.py
import os
import shutil
import torch
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset

# --- PATH CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', '03_processed')
MODEL_OUTPUT_DIR = os.path.join(BASE_DIR, 'models', 'stage1_dapt_distilbert')
TRAIN_FILE = os.path.join(PROCESSED_DIR, 'hybrid_corpus.txt')

# --- OPTIMIZED HYPERPARAMETERS ---
MODEL_CHECKPOINT = "distilbert-base-multilingual-cased"
BATCH_SIZE = 16

# CRITICAL SETTINGS FOR DAPT > BASE
LEARNING_RATE = 2e-5  
EPOCHS = 30  # INCREASED: 6 -> 30. This ensures the domain knowledge sticks.
MLM_PROBABILITY = 0.15

def main():
    print(f"Project Root Detected: {BASE_DIR}")
    
    if not os.path.exists(TRAIN_FILE):
        print(f"[ERROR] Training data not found at: {TRAIN_FILE}")
        return

    print("--- STAGE 1: Domain-Adaptive Pre-training (DAPT) ---")
    
    # Clean previous run to ensure fresh training
    if os.path.exists(MODEL_OUTPUT_DIR):
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
        fp16=torch.cuda.is_available(),
        logging_dir=os.path.join(BASE_DIR, 'logs', 'dapt'),
        report_to="none"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        data_collator=data_collator,
    )
    
    trainer.train()
    
    print("Saving DAPT model...")
    trainer.save_model(MODEL_OUTPUT_DIR)
    tokenizer.save_pretrained(MODEL_OUTPUT_DIR)
    print(f"SUCCESS: DAPT Model saved to {MODEL_OUTPUT_DIR}")

if __name__ == "__main__":
    main()