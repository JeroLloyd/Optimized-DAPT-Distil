import os
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import Dataset
from sklearn.metrics import f1_score, confusion_matrix
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, 
    TrainingArguments, Trainer, set_seed
)

# --- PATH CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(SCRIPT_DIR)
BASE_DIR = os.path.dirname(SRC_DIR)
FIRECS_DIR = os.path.join(BASE_DIR, 'data', '03_processed', 'FiReCS_Final')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
FIG_DIR = os.path.join(BASE_DIR, 'reports', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

MODEL_A_PATH = os.path.join(MODELS_DIR, "model_a_base")
MODEL_B_PATH = os.path.join(MODELS_DIR, "model_b_dapt")
BASE_TOKENIZER = "distilbert-base-multilingual-cased"

def get_predictions(model_path, dataset):
    tokenizer = AutoTokenizer.from_pretrained(BASE_TOKENIZER)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, 
        problem_type="single_label_classification"
    )
    
    def tokenize(batch):
        return tokenizer(batch['review'], truncation=True, max_length=128, padding='max_length')
    
    tokenized_ds = dataset.map(tokenize, batched=True)
    trainer = Trainer(model=model)
    preds = trainer.predict(tokenized_ds)
    y_pred = np.argmax(preds.predictions, axis=-1)
    return y_pred, preds.label_ids

def main():
    print("=== PART 1: GENERATING COMPARATIVE CONFUSION MATRIX ===")
    test_df = pd.read_csv(os.path.join(FIRECS_DIR, 'test.csv'))
    test_df['label'] = test_df['label'].astype(int)
    test_ds = Dataset.from_pandas(test_df)
    
    print("Evaluating Model A (Generic Baseline)...")
    y_pred_a, y_true = get_predictions(MODEL_A_PATH, test_ds)
    
    print("Evaluating Model B (DAPT Baseline)...")
    y_pred_b, _ = get_predictions(MODEL_B_PATH, test_ds)
    
    labels = ['Negative', 'Neutral', 'Positive']
    cm_a = confusion_matrix(y_true, y_pred_a)
    cm_b = confusion_matrix(y_true, y_pred_b)

    # --- EXPORT CONFUSION MATRICES TO CSV ---
    CM_DIR = os.path.join(FIG_DIR, '07_cm_comparison')
    os.makedirs(CM_DIR, exist_ok=True)
    
    cm_a_df = pd.DataFrame(cm_a, index=labels, columns=[f"Pred_{l}" for l in labels])
    cm_a_csv_path = os.path.join(CM_DIR, 'cm_comparison_model_a.csv')
    cm_a_df.to_csv(cm_a_csv_path)
    
    cm_b_df = pd.DataFrame(cm_b, index=labels, columns=[f"Pred_{l}" for l in labels])
    cm_b_csv_path = os.path.join(CM_DIR, 'cm_comparison_model_b.csv')
    cm_b_df.to_csv(cm_b_csv_path)
    
    print(f"SUCCESS: Exported Model A CSV to {cm_a_csv_path}")
    print(f"SUCCESS: Exported Model B CSV to {cm_b_csv_path}")
    
    # Plot Side-by-Side CM
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    sns.heatmap(cm_a, annot=True, fmt='d', cmap='Blues', ax=axes[0], xticklabels=labels, yticklabels=labels)
    axes[0].set_title('Generic DistilmBERT (Model A)', fontweight='bold', pad=15)
    axes[0].set_ylabel('True Sentiment', fontweight='bold')
    axes[0].set_xlabel('Predicted Sentiment', fontweight='bold')
    
    sns.heatmap(cm_b, annot=True, fmt='d', cmap='Greens', ax=axes[1], xticklabels=labels, yticklabels=labels)
    axes[1].set_title('DAPT-DistilmBERT (Model B)', fontweight='bold', pad=15)
    axes[1].set_xlabel('Predicted Sentiment', fontweight='bold')
    
    plt.tight_layout()
    cm_path = os.path.join(CM_DIR, 'cm_comparison.png')
    plt.savefig(cm_path, dpi=300)
    print(f"SUCCESS: Comparative CM saved to {cm_path}\n")

    print("=== PART 2: MULTI-SEED VARIANCE TEST (STANDARD DEVIATION) ===")
    seeds = [42, 123, 777]
    results_b = []

    train_df = pd.read_csv(os.path.join(FIRECS_DIR, 'train.csv')).rename(columns={'review':'text'})
    val_df = pd.read_csv(os.path.join(FIRECS_DIR, 'val.csv')).rename(columns={'review':'text'})
    
    train_df['label'] = train_df['label'].astype(int)
    val_df['label'] = val_df['label'].astype(int)
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_TOKENIZER)
    
    def tokenize_fn(x):
        return tokenizer(x["text"], truncation=True, max_length=128, padding='max_length')

    train_ds = Dataset.from_pandas(train_df).map(tokenize_fn, batched=True)
    val_ds = Dataset.from_pandas(val_df).map(tokenize_fn, batched=True)
    
    for seed in seeds:
        print(f"\nRunning DAPT Fine-Tuning with Seed: {seed}...")
        set_seed(seed)
        
        model = AutoModelForSequenceClassification.from_pretrained(
            os.path.join(MODELS_DIR, 'stage1_dapt_distilbert'), 
            num_labels=3,
            problem_type="single_label_classification"
        )
        
        args = TrainingArguments(
            output_dir=f"./temp_seed_{seed}", num_train_epochs=8, learning_rate=2e-5, 
            per_device_train_batch_size=32, evaluation_strategy="epoch", save_strategy="no", report_to="none"
        )
        
        def compute_metrics(eval_pred):
            preds = np.argmax(eval_pred.predictions, axis=-1)
            return {"macro_f1": f1_score(eval_pred.label_ids, preds, average='macro')}
            
        trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds, compute_metrics=compute_metrics)
        trainer.train()
        
        eval_res = trainer.evaluate()
        f1 = eval_res['eval_macro_f1']
        results_b.append(f1)
        print(f"Seed {seed} Macro F1: {f1:.4f}")

    mean_f1 = np.mean(results_b)
    std_f1 = np.std(results_b)
    print("\n=== STATISTICAL VALIDATION RESULTS ===")
    print(f"DAPT Model Mean Macro F1: {mean_f1:.4f}")
    print(f"DAPT Model Standard Deviation: ±{std_f1:.4f}")

if __name__ == "__main__":
    main()