from transformers import (
    AutoTokenizer, AutoModelForMaskedLM, 
    DataCollatorForLanguageModeling, Trainer, TrainingArguments
)
from datasets import load_dataset
import sys
import os
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

def train_stage1():
    print("\n" + "="*50)
    print("[STAGE 1] Domain-Adaptive Pre-Training (DAPT)...")
    print("Goal: Adapting Model B to Taglish E-commerce Domain")
    print("="*50)
    
    config.set_seed()

    # 1. Load the Lazada Corpus
    dataset = load_dataset('text', data_files={'train': config.DAPT_CORPUS_PATH})
    
    model_checkpoint = "distilbert-base-multilingual-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    # 2. Tokenization with specific length for DAPT
    def tokenize_func(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=config.MAX_LEN)

    tokenized_datasets = dataset.map(tokenize_func, batched=True, remove_columns=["text"])

    # 3. Masked Language Modeling (MLM) Setup
    model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=config.MLM_PROBABILITY
    )

    # 4. Optimized Training Arguments for DAPT
    # We use a lower Learning Rate and Weight Decay for fairness and stability
    args = TrainingArguments(
    output_dir=config.MODEL_B_BASE_DIR,
    per_device_train_batch_size=config.BATCH_SIZE,
    num_train_epochs=config.EPOCHS_DAPT + 2, # Increase to allow knowledge to settle
    learning_rate=5e-5,                     # Lower LR for stable pre-training
    lr_scheduler_type="cosine",              # Smoother decay prevents shock
    warmup_ratio=0.1,                        # Stable start
    weight_decay=0.01,                       # Regularization
    save_strategy="epoch",
    seed=config.SEED,
    fp16=torch.cuda.is_available()
)

    # 5. Trainer Initialization
    trainer = Trainer(
        model=model, 
        args=args, 
        train_dataset=tokenized_datasets["train"], 
        data_collator=data_collator
    )

    # 6. Execution
    print("Starting DAPT training... This will optimize the model for Taglish nuances.")
    trainer.train()
    
    # 7. Save the DAPT Base Model
    trainer.save_model(config.MODEL_B_BASE_DIR)
    tokenizer.save_pretrained(config.MODEL_B_BASE_DIR)
    print(f"\n[SUCCESS] DAPT Model saved to {config.MODEL_B_BASE_DIR}")

if __name__ == "__main__":
    train_stage1()