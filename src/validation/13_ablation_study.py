# FILE: 13_ablation_study.py
import os
import shutil
import pandas as pd
import torch
import time
import numpy as np
import gc
from datasets import load_dataset, Dataset, DatasetDict
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling,
    AutoModelForSequenceClassification, DataCollatorWithPadding, 
    TrainingArguments, Trainer, EarlyStoppingCallback, set_seed
)

# --- DETERMINISTIC SEED LOCK ---
def lock_environmental_seeds(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    set_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

lock_environmental_seeds(42)

# --- PATH CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(SCRIPT_DIR)
BASE_DIR = os.path.dirname(SRC_DIR)
INTERIM_DIR = os.path.join(BASE_DIR, 'data', '02_interim')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', '03_processed')
FIRECS_DIR = os.path.join(PROCESSED_DIR, 'FiReCS_Final')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports', 'metrics')

AUTH_TXT = os.path.join(PROCESSED_DIR, 'authentic_only.txt')
DAPT_OUT = os.path.join(BASE_DIR, 'models', 'ablation_dapt_authentic')
FT_OUT = os.path.join(BASE_DIR, 'models', 'ablation_finetuned')

os.makedirs(REPORTS_DIR, exist_ok=True)

# SYNCED SEARCH RATES FROM REFINED STAGE 2
UNIVERSAL_SEARCH_RATES = [1e-5, 1.2e-5, 1.5e-5]

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='macro')
    return {"accuracy": acc, "macro_f1": f1}

def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# --- EQUITABLE LLRD LOGIC (Matches Stage 2 perfectly) ---
def get_optimizer_grouped_parameters(model, base_lr, decay_factor=0.95, weight_decay=0.01):
    """Assigns decaying learning rates to model layers."""
    is_distilbert = hasattr(model, "distilbert")
    layers = model.distilbert.transformer.layer if is_distilbert else model.roberta.encoder.layer
    embeddings = model.distilbert.embeddings if is_distilbert else model.roberta.embeddings
    num_layers = len(layers)
    
    params = []
    
    # 1. Head (Full LR)
    params.append({
        "params": [p for n, p in model.named_parameters() if "classifier" in n or "pre_classifier" in n],
        "weight_decay": weight_decay,
        "lr": base_lr,
    })
    
    # 2. Layers (Decaying)
    for i in range(num_layers - 1, -1, -1):
        params.append({
            "params": layers[i].parameters(),
            "weight_decay": weight_decay,
            "lr": base_lr * (decay_factor ** (num_layers - i)),
        })
        
    # 3. Embeddings (Protected)
    params.append({
        "params": embeddings.parameters(),
        "weight_decay": weight_decay,
        "lr": base_lr * (decay_factor ** (num_layers + 1)),
    })
    return params

def main():
    print("=== CHECKING DAPT PROGRESS ===")
    if not os.path.exists(DAPT_OUT) or not os.listdir(DAPT_OUT):
        print("\n=== STEP 1: ISOLATING AUTHENTIC DATA ===")
        lazada_csv = os.path.join(INTERIM_DIR, "cleaned_lazada_data.csv")
        df_auth = pd.read_csv(lazada_csv)
        
        with open(AUTH_TXT, 'w', encoding='utf-8') as f:
            for text in df_auth['final_text'].dropna():
                f.write(text + "\n")
        print(f"Isolated {len(df_auth)} authentic samples.")

        print("\n=== STEP 2: DAPT ON AUTHENTIC DATA ONLY (SCIENTIFIC SYNC) ===")
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")
        dataset = load_dataset('text', data_files={'train': AUTH_TXT})
        
        tokenized_dapt = dataset.map(
            lambda x: tokenizer(x["text"], truncation=True, max_length=128), 
            batched=True, remove_columns=["text"]
        )
        
        dapt_model = AutoModelForMaskedLM.from_pretrained("distilbert-base-multilingual-cased")
        
        # SYNCED WITH STAGE 1 (10 Epochs, LR 3e-5, Accumulation 2)
        dapt_args = TrainingArguments(
            output_dir=DAPT_OUT, 
            num_train_epochs=10, 
            per_device_train_batch_size=16,
            gradient_accumulation_steps=2,
            learning_rate=3e-5,    
            weight_decay=0.01,
            warmup_ratio=0.15,
            save_strategy="no", 
            report_to="none",
            fp16=torch.cuda.is_available()
        )
        
        Trainer(
            model=dapt_model, args=dapt_args, train_dataset=tokenized_dapt["train"],
            data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
        ).train()
        
        dapt_model.save_pretrained(DAPT_OUT)
        tokenizer.save_pretrained(DAPT_OUT)
        clear_memory()
    else:
        print(f"[SKIP] DAPT already completed. Model found at: {DAPT_OUT}")

    print("\n=== STEP 3: AUTOMATED FINE-TUNING GRID SEARCH (ABLATION) ===")
    tokenizer = AutoTokenizer.from_pretrained(DAPT_OUT)
    
    train_df = pd.read_csv(os.path.join(FIRECS_DIR, 'train.csv'))
    val_df = pd.read_csv(os.path.join(FIRECS_DIR, 'val.csv'))
    train_df['label'] = train_df['label'].astype(int)
    val_df['label'] = val_df['label'].astype(int)
    
    raw_datasets = DatasetDict({
        "train": Dataset.from_pandas(train_df[['review', 'label']].rename(columns={'review':'text'})),
        "validation": Dataset.from_pandas(val_df[['review', 'label']].rename(columns={'review':'text'}))
    })
    
    tokenized_ft = raw_datasets.map(
        lambda x: tokenizer(x["text"], truncation=True, max_length=128), batched=True
    )
    
    best_f1 = 0.0
    best_lr = None
    best_temp_dir = ""

    for lr in UNIVERSAL_SEARCH_RATES:
        print(f"\n[TESTING ABLATION] LR: {lr} | Applying Precision LLRD (0.95)")
        temp_dir = f"{FT_OUT}_TEMP_LR_{lr}"
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        ft_model = AutoModelForSequenceClassification.from_pretrained(
            DAPT_OUT, num_labels=3, problem_type="single_label_classification"
        )
        
        # SYNCED OPTIMIZER: 0.95 factor, 0.01 weight decay
        optimizer = torch.optim.AdamW(get_optimizer_grouped_parameters(ft_model, lr, decay_factor=0.95, weight_decay=0.01), lr=lr)
        
        # SYNCED FT ARGS: Batch size 16 to match Stage 2 exactly
        ft_args = TrainingArguments(
            output_dir=temp_dir, 
            evaluation_strategy="epoch", 
            save_strategy="epoch",
            learning_rate=lr, 
            per_device_train_batch_size=16, 
            gradient_accumulation_steps=1,
            num_train_epochs=15,
            weight_decay=0.01,
            warmup_ratio=0.10,
            load_best_model_at_end=True, 
            metric_for_best_model="macro_f1", 
            greater_is_better=True,
            report_to="none",
            fp16=torch.cuda.is_available(),
            label_smoothing_factor=0.1
        )
        
        trainer = Trainer(
            model=ft_model, args=ft_args, train_dataset=tokenized_ft["train"],
            eval_dataset=tokenized_ft["validation"], data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
            compute_metrics=compute_metrics, 
            optimizers=(optimizer, None),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)] # SYNCED: Patience 2
        )
        
        trainer.train()
        trainer.save_model(temp_dir)
        tokenizer.save_pretrained(temp_dir)
        
        eval_metrics = trainer.evaluate()
        current_f1 = eval_metrics['eval_macro_f1']
        
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_lr = lr
            best_temp_dir = temp_dir
            
        del ft_model, trainer
        clear_memory()

    print(f"\n[WINNER] Best Ablation LR: {best_lr} (F1: {best_f1:.4f})")
    if os.path.exists(FT_OUT): shutil.rmtree(FT_OUT)
    shutil.copytree(best_temp_dir, FT_OUT)
    
    for lr in UNIVERSAL_SEARCH_RATES:
        td = f"{FT_OUT}_TEMP_LR_{lr}"
        if os.path.exists(td): shutil.rmtree(td)

    print("\n=== STEP 4: CONSISTENT TEST SET EVALUATION ===")
    test_df = pd.read_csv(os.path.join(FIRECS_DIR, 'test.csv'))
    test_df['label'] = test_df['label'].astype(int)
    test_ds = Dataset.from_pandas(test_df[['review', 'label']].rename(columns={'review':'text'}))
    tokenized_test = test_ds.map(lambda x: tokenizer(x["text"], truncation=True, max_length=128), batched=True)
    
    final_model = AutoModelForSequenceClassification.from_pretrained(FT_OUT)
    final_trainer = Trainer(
        model=final_model, tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics
    )
    
    test_metrics = final_trainer.evaluate(tokenized_test)
    macro_f1_test = test_metrics['eval_macro_f1']
    accuracy_test = test_metrics['eval_accuracy']
    
    results_df = pd.DataFrame([{
        "Model": "Authentic-Only DAPT (Ablation)",
        "Macro F1 Score": round(macro_f1_test, 4),
        "Accuracy": round(accuracy_test, 4),
        "Best_Learning_Rate": best_lr,
        "Data_Source": "Official Test Set"
    }])
    
    csv_path = os.path.join(REPORTS_DIR, 'ablation_metrics.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"Ablation metrics saved to: {csv_path}")

if __name__ == "__main__":
    main()