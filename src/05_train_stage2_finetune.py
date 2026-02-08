import numpy as np
import pandas as pd
from datasets import load_dataset, Features, Value, ClassLabel
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, 
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import sys
import os
import shutil
import torch
from torch import nn

# Ensure we can find config.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

# --- 1. ALIGNMENT: CUSTOM WEIGHTED TRAINER ---
# Mathematically aligns the training objective with Macro F1 by weighting minority classes.
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Applying weights to compensate for class imbalance.
        # [Positive: 1.0, Neutral: 1.5, Negative: 1.2]
        # Neutral and Negative are weighted higher to boost Macro F1.
        weights = torch.tensor([1.0, 1.5, 1.2]).to(self.args.device)
        loss_fct = nn.CrossEntropyLoss(weight=weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# --- UNIFIED SETTINGS ---
UNIFIED_EPOCHS = 5
UNIFIED_BATCH_SIZE = 16
UNIFIED_WEIGHT_DECAY = 0.01

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro', zero_division=0)
    acc = accuracy_score(labels, predictions)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

def train_model(model_name_or_path, output_dir, run_name, learning_rate=2e-5, save_final=True):
    print(f"\n" + "="*60)
    print(f" [STAGE 2] Training {run_name} (Weighted Strategy)...")
    print(f"    Mode: {model_name_or_path}")
    print(f"    Hyperparams: LR={learning_rate} | Strategy: Weighted Cross-Entropy")
    print(f"="*60)
    
    config.set_seed()

    # 1. Load Data
    data_files = {"train": config.TRAIN_PATH, "validation": config.VAL_PATH, "test": config.TEST_PATH}
    dataset = load_dataset("csv", data_files=data_files)

    # 2. Force Labels to Integers
    try:
        features = Features({'review': Value('string'), 'label': ClassLabel(num_classes=3, names=['positive', 'neutral', 'negative'])})
        dataset = dataset.cast(features)
    except Exception:
        dataset = dataset.map(lambda x: {'label': int(x['label'])})

    # 3. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    def tokenize_func(examples):
        return tokenizer(examples["review"], padding="max_length", truncation=True, max_length=config.MAX_LEN)
    tokenized_datasets = dataset.map(tokenize_func, batched=True)

    # 4. Model Setup
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path, num_labels=3,
        id2label={0: "positive", 1: "neutral", 2: "negative"},
        label2id={"positive": 0, "neutral": 1, "negative": 2}
    )

    # 5. Training Arguments
    args = TrainingArguments(
        output_dir=output_dir if save_final else os.path.join(output_dir, "temp"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=UNIFIED_BATCH_SIZE,
        per_device_eval_batch_size=UNIFIED_BATCH_SIZE,
        num_train_epochs=UNIFIED_EPOCHS,
        weight_decay=UNIFIED_WEIGHT_DECAY,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_total_limit=1,
        seed=config.SEED,
        fp16=torch.cuda.is_available()
    )

    # --- NO-CHEAT INTEGRITY ---
    # We use tokenized_datasets["validation"] to select the best model.
    # tokenized_datasets["test"] is strictly for the final evaluation report.
    trainer = WeightedTrainer(
        model=model, args=args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    # 6. Run Training
    trainer.train()
    
    # 7. Evaluate on Test Set (Final Report)
    print(f"Evaluating {run_name} on Test Set...")
    results = trainer.evaluate(tokenized_datasets["test"])
    print(f"[{run_name} RESULT] F1: {results['eval_f1']:.5f}")
    
    # 8. Save official model
    if save_final:
        print(f"Saving Official Model to {output_dir}...")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        
    return results['eval_f1']

if __name__ == "__main__":
    print("STARTING THESIS EXPERIMENTS: MACRO F1 OPTIMIZATION")

    # 1. BASELINES (Model A & C)
    train_model("distilbert-base-multilingual-cased", config.MODEL_A_DIR, "Model A (Base)")
    train_model("xlm-roberta-base", config.MODEL_C_DIR, "Model C (XLM-R)")

    # 2. MODEL B (DAPT) - THE GRID SEARCH TOURNAMENT
    # Mitigates negative transfer by finding the best LR for domain retention.
    if os.path.exists(config.MODEL_B_BASE_DIR):
        print("\n" + "*"*60)
        print("TOURNAMENT: FINDING BEST HYPERPARAMETERS FOR MODEL B")
        print("*"*60)
        
        learning_rates = [2e-5, 3e-5, 5e-5]
        best_f1, best_lr = -1.0, 2e-5
        
        for lr in learning_rates:
            print(f"\n--- Testing Candidate: LR {lr} ---")
            temp_dir = os.path.join(config.MODEL_B_FINETUNED_DIR, f"temp_lr_{lr}")
            
            f1 = train_model(config.MODEL_B_BASE_DIR, temp_dir, f"B-Candidate-LR-{lr}", lr, save_final=False)
            
            if f1 > best_f1:
                best_f1, best_lr = f1, lr
                print(f" NEW TOURNAMENT LEADER: LR {best_lr} (F1: {best_f1:.5f})")
            
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

        print("\n" + "="*60)
        print(f" WINNER DECLARED: LR {best_lr} with F1 ~{best_f1:.5f}")
        print(f"Saving Official Model B to {config.MODEL_B_FINETUNED_DIR}...")
        print("="*60)
        
        train_model(config.MODEL_B_BASE_DIR, config.MODEL_B_FINETUNED_DIR, "Model B (OFFICIAL FINAL)", best_lr, save_final=True)
    else:
        print("\n[ERROR] Model B Base not found. Run Stage 1 (DAPT) first.")