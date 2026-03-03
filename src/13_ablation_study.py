import os
import pandas as pd
import torch
import time
import numpy as np
from datasets import load_dataset, Dataset, DatasetDict
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling,
    AutoModelForSequenceClassification, DataCollatorWithPadding, 
    TrainingArguments, Trainer, EarlyStoppingCallback, set_seed
)

# --- PATHS ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
INTERIM_DIR = os.path.join(BASE_DIR, 'data', '02_interim')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', '03_processed')
FIRECS_DIR = os.path.join(PROCESSED_DIR, 'FiReCS_Final')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports', 'metrics')

AUTH_TXT = os.path.join(PROCESSED_DIR, 'authentic_only.txt')
DAPT_OUT = os.path.join(BASE_DIR, 'models', 'ablation_dapt_authentic')
FT_OUT = os.path.join(BASE_DIR, 'models', 'ablation_finetuned')

os.makedirs(REPORTS_DIR, exist_ok=True)
set_seed(42)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='macro')
    return {"accuracy": acc, "macro_f1": f1}

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

        print("\n=== STEP 2: DAPT ON AUTHENTIC DATA ONLY ===")
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")
        dataset = load_dataset('text', data_files={'train': AUTH_TXT})
        
        tokenized_dapt = dataset.map(
            lambda x: tokenizer(x["text"], truncation=True, max_length=128), 
            batched=True, remove_columns=["text"]
        )
        
        dapt_model = AutoModelForMaskedLM.from_pretrained("distilbert-base-multilingual-cased")
        dapt_args = TrainingArguments(
            output_dir=DAPT_OUT, num_train_epochs=30, per_device_train_batch_size=16,
            learning_rate=2e-5, weight_decay=0.01, warmup_ratio=0.06, 
            save_strategy="no", report_to="none"
        )
        
        Trainer(
            model=dapt_model, args=dapt_args, train_dataset=tokenized_dapt["train"],
            data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
        ).train()
        
        dapt_model.save_pretrained(DAPT_OUT)
        tokenizer.save_pretrained(DAPT_OUT)
    else:
        print(f"[SKIP] DAPT already completed. Model found at: {DAPT_OUT}")

    print("\n=== STEP 3: FINE-TUNING ABLATION MODEL ===")
    tokenizer = AutoTokenizer.from_pretrained(DAPT_OUT if os.path.exists(DAPT_OUT) else "distilbert-base-multilingual-cased")
    
    train_df = pd.read_csv(os.path.join(FIRECS_DIR, 'train.csv'))
    val_df = pd.read_csv(os.path.join(FIRECS_DIR, 'val.csv'))
    
    # FIX: Explicitly enforce integer types for classification
    train_df['label'] = train_df['label'].astype(int)
    val_df['label'] = val_df['label'].astype(int)
    
    raw_datasets = DatasetDict({
        "train": Dataset.from_pandas(train_df[['review', 'label']].rename(columns={'review':'text'})),
        "validation": Dataset.from_pandas(val_df[['review', 'label']].rename(columns={'review':'text'}))
    })
    
    tokenized_ft = raw_datasets.map(
        lambda x: tokenizer(x["text"], truncation=True, max_length=128), batched=True
    )
    
    # FIX: Explicitly define problem_type to prevent BCE error
    ft_model = AutoModelForSequenceClassification.from_pretrained(
        DAPT_OUT, 
        num_labels=3,
        problem_type="single_label_classification"
    )
    
    ft_args = TrainingArguments(
        output_dir=FT_OUT, evaluation_strategy="epoch", save_strategy="epoch",
        learning_rate=2e-5, per_device_train_batch_size=32, num_train_epochs=8,
        load_best_model_at_end=True, metric_for_best_model="macro_f1", report_to="none"
    )
    
    ft_trainer = Trainer(
        model=ft_model, args=ft_args, train_dataset=tokenized_ft["train"],
        eval_dataset=tokenized_ft["validation"], data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics, callbacks=[EarlyStoppingCallback(early_stopping_patience=4)]
    )
    
    ft_trainer.train()
    
    print("\n=== ABLATION RESULTS ===")
    metrics = ft_trainer.evaluate()
    macro_f1 = metrics['eval_macro_f1']
    accuracy = metrics['eval_accuracy']
    
    print(f"Authentic-Only DAPT Macro F1-Score: {macro_f1:.4f}")
    print("Compare this score to your Hybrid F1-Score (0.8147) in your manuscript.")

    # SAVE METRICS TO CSV
    results_df = pd.DataFrame([{
        "Model": "Authentic-Only DAPT (Ablation)",
        "Macro F1 Score": macro_f1,
        "Accuracy": accuracy,
        "Evaluation Loss": metrics['eval_loss']
    }])
    
    csv_path = os.path.join(REPORTS_DIR, 'ablation_metrics.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"Metrics successfully saved to: {csv_path}")

if __name__ == "__main__":
    main()