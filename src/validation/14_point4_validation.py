# FILE: 14_point4_validation.py
import os
import shutil
import pandas as pd
import numpy as np
import torch
import gc
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import Dataset
from sklearn.metrics import f1_score, confusion_matrix
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, 
    TrainingArguments, Trainer, set_seed, EarlyStoppingCallback,
    DataCollatorWithPadding
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
FIRECS_DIR = os.path.join(BASE_DIR, 'data', '03_processed', 'FiReCS_Final')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

FIG_DIR = os.path.join(BASE_DIR, 'reports', 'figures')
METRICS_DIR = os.path.join(BASE_DIR, 'reports', 'metrics')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

MODEL_A_PATH = os.path.join(MODELS_DIR, "model_a_base")
MODEL_B_PATH = os.path.join(MODELS_DIR, "model_b_dapt")
DAPT_BASE_PATH = os.path.join(MODELS_DIR, 'stage1_dapt_distilmbert') 
BASE_TOKENIZER = "distilbert-base-multilingual-cased"

def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# --- SYNCED: LLRD Optimizer Logic from Stage 2 ---
def get_optimizer_grouped_parameters(model, base_lr, weight_decay=0.01):
    """Assigns decaying learning rates to model layers from top to bottom."""
    decay_factor = 0.95
    layers = model.distilbert.transformer.layer
    embeddings = model.distilbert.embeddings
    num_layers = len(layers)
    
    optimizer_grouped_parameters = []
    
    # 1. Classifier Head (Full LR)
    optimizer_grouped_parameters.append({
        "params": [p for n, p in model.named_parameters() if "classifier" in n or "pre_classifier" in n],
        "weight_decay": weight_decay,
        "lr": base_lr,
    })
    
    # 2. Transformer Layers (Decaying LR)
    for i in range(num_layers - 1, -1, -1):
        lr = base_lr * (decay_factor ** (num_layers - i))
        optimizer_grouped_parameters.append({
            "params": layers[i].parameters(),
            "weight_decay": weight_decay,
            "lr": lr,
        })
        
    # 3. Embeddings (Lowest LR)
    lr_embed = base_lr * (decay_factor ** (num_layers + 1))
    optimizer_grouped_parameters.append({
        "params": embeddings.parameters(),
        "weight_decay": weight_decay,
        "lr": lr_embed,
    })
    return optimizer_grouped_parameters

def get_predictions(model_path, dataset):
    tokenizer = AutoTokenizer.from_pretrained(BASE_TOKENIZER)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, problem_type="single_label_classification"
    )
    
    def tokenize(batch):
        return tokenizer(batch['review'], truncation=True, max_length=128)
    
    tokenized_ds = dataset.map(tokenize, batched=True)
    
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
    )
    
    preds = trainer.predict(tokenized_ds)
    y_pred = np.argmax(preds.predictions, axis=-1)
    
    del model, trainer
    clear_memory()
    return y_pred, preds.label_ids

def main():
    print("=== PART 1: GENERATING COMPARATIVE CONFUSION MATRIX ===")
    test_df = pd.read_csv(os.path.join(FIRECS_DIR, 'test.csv'))
    test_df['label'] = test_df['label'].astype(int)
    test_ds_raw = Dataset.from_pandas(test_df)
    
    y_pred_a, y_true = get_predictions(MODEL_A_PATH, test_ds_raw)
    y_pred_b, _ = get_predictions(MODEL_B_PATH, test_ds_raw)
    
    labels = ['Negative', 'Neutral', 'Positive']
    cm_a = confusion_matrix(y_true, y_pred_a)
    cm_b = confusion_matrix(y_true, y_pred_b)

    CM_DIR = os.path.join(FIG_DIR, '07_cm_comparison')
    os.makedirs(CM_DIR, exist_ok=True)
    
    pd.DataFrame(cm_a, index=labels, columns=[f"Pred_{l}" for l in labels]).to_csv(os.path.join(CM_DIR, 'cm_comparison_model_a.csv'))
    pd.DataFrame(cm_b, index=labels, columns=[f"Pred_{l}" for l in labels]).to_csv(os.path.join(CM_DIR, 'cm_comparison_model_b.csv'))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.heatmap(cm_a, annot=True, fmt='d', cmap='Blues', ax=axes[0], xticklabels=labels, yticklabels=labels)
    axes[0].set_title('Generic DistilmBERT (Model A)', fontweight='bold', pad=15)
    sns.heatmap(cm_b, annot=True, fmt='d', cmap='Greens', ax=axes[1], xticklabels=labels, yticklabels=labels)
    axes[1].set_title('DAPT-DistilmBERT (Model B)', fontweight='bold', pad=15)
    plt.savefig(os.path.join(CM_DIR, 'cm_comparison.png'), dpi=300)
    plt.close()

    print("\n=== PART 2: AUTOMATED MULTI-SEED VARIANCE TEST (EQUITABLE SYNC) ===")
    
    BEST_LR = 1e-5 
    summary_path = os.path.join(METRICS_DIR, "finetuning_grid_search_results.csv")
    if os.path.exists(summary_path):
        results_df = pd.read_csv(summary_path)
        model_b_data = results_df[results_df['Model_Name'].str.contains("Model B")]
        if not model_b_data.empty:
            BEST_LR = model_b_data.loc[model_b_data['Macro_F1'].idxmax(), 'Learning_Rate']
            print(f"[AUTO] Using Best LR for Model B: {BEST_LR}")

    seeds = [42, 123, 777]
    results_b = []
    detailed_csv_data = []

    train_df = pd.read_csv(os.path.join(FIRECS_DIR, 'train.csv')).rename(columns={'review':'text'})
    val_df = pd.read_csv(os.path.join(FIRECS_DIR, 'val.csv')).rename(columns={'review':'text'})
    test_df = pd.read_csv(os.path.join(FIRECS_DIR, 'test.csv')).rename(columns={'review':'text'})
    train_df['label'], val_df['label'], test_df['label'] = train_df['label'].astype(int), val_df['label'].astype(int), test_df['label'].astype(int)
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_TOKENIZER)
    def tokenize_fn(x): return tokenizer(x["text"], truncation=True, max_length=128)

    train_ds = Dataset.from_pandas(train_df).map(tokenize_fn, batched=True)
    val_ds = Dataset.from_pandas(val_df).map(tokenize_fn, batched=True)
    test_ds = Dataset.from_pandas(test_df).map(tokenize_fn, batched=True)
    
    def compute_metrics(eval_pred):
        preds = np.argmax(eval_pred.predictions, axis=-1)
        return {"macro_f1": f1_score(eval_pred.label_ids, preds, average='macro')}

    for seed in seeds:
        print(f"\nEvaluating Seed: {seed}...")
        lock_environmental_seeds(seed)
        
        if seed == 42:
            model = AutoModelForSequenceClassification.from_pretrained(MODEL_B_PATH)
            trainer = Trainer(
                model=model, tokenizer=tokenizer,
                data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
                compute_metrics=compute_metrics
            )
            eval_res = trainer.evaluate(eval_dataset=test_ds)
        else:
            print(f"[INFO] Training Seed {seed} from scratch using LR {BEST_LR} and Equitable LLRD...")
            model = AutoModelForSequenceClassification.from_pretrained(DAPT_BASE_PATH, num_labels=3)
            
            grouped_params = get_optimizer_grouped_parameters(model, BEST_LR, weight_decay=0.01)
            optimizer = torch.optim.AdamW(grouped_params)
            
            # CRITICAL FIX: Match Stage 2 academic parameters strictly
            args = TrainingArguments(
                output_dir=f"./temp_seed_{seed}", num_train_epochs=15, 
                learning_rate=BEST_LR, per_device_train_batch_size=16, # SYNCED
                evaluation_strategy="epoch", save_strategy="epoch", 
                load_best_model_at_end=True, metric_for_best_model="macro_f1", 
                greater_is_better=True, report_to="none",
                label_smoothing_factor=0.1, warmup_ratio=0.10, weight_decay=0.01 # SYNCED
            )
            
            trainer = Trainer(
                model=model, args=args, optimizers=(optimizer, None),
                train_dataset=train_ds, eval_dataset=val_ds, tokenizer=tokenizer,
                data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
                compute_metrics=compute_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=2)] # SYNCED
            )
            trainer.train()
            eval_res = trainer.evaluate(eval_dataset=test_ds)
            if os.path.exists(f"./temp_seed_{seed}"): shutil.rmtree(f"./temp_seed_{seed}", ignore_errors=True)

        f1 = eval_res['eval_macro_f1']
        results_b.append(f1)
        detailed_csv_data.append({"Evaluation_Type": f"Seed {seed}", "Macro_F1_Score": round(f1, 4)})
        del model, trainer
        clear_memory()

    mean_f1, std_f1 = np.mean(results_b), np.std(results_b)
    detailed_csv_data.extend([{"Evaluation_Type": "Mean", "Macro_F1_Score": round(mean_f1, 4)},
                              {"Evaluation_Type": "Standard Deviation", "Macro_F1_Score": round(std_f1, 4)}])
    
    pd.DataFrame(detailed_csv_data).to_csv(os.path.join(METRICS_DIR, 'multi_seed_variance_results.csv'), index=False)
    print(f"\n[SUCCESS] Variance Results: {mean_f1:.4f} (±{std_f1:.4f})")

if __name__ == "__main__":
    main()