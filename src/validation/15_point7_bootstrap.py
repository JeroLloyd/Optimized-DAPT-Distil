# FILE: 15_point7_bootstrap.py
import os
import pandas as pd
import numpy as np
import torch
import gc
from datasets import Dataset
from sklearn.metrics import f1_score
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    set_seed,
    DataCollatorWithPadding
)

# --- PATH CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(SCRIPT_DIR)
BASE_DIR = os.path.dirname(SRC_DIR)
FIRECS_DIR = os.path.join(BASE_DIR, 'data', '03_processed', 'FiReCS_Final')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports', 'metrics')

os.makedirs(REPORTS_DIR, exist_ok=True)

MODEL_A_PATH = os.path.join(MODELS_DIR, "model_a_base")
MODEL_B_PATH = os.path.join(MODELS_DIR, "model_b_dapt")
BASE_TOKENIZER = "distilbert-base-multilingual-cased"

# Use seed 42 for consistency with the rest of the project
set_seed(42)

def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def get_predictions(model_path, dataset):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
        
    tokenizer = AutoTokenizer.from_pretrained(BASE_TOKENIZER)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, 
        problem_type="single_label_classification"
    )
    
    # SYNCED: Tokenization without hard padding to allow dynamic collator control
    def tokenize(batch):
        return tokenizer(batch['text'], truncation=True, max_length=128)
    
    tokenized_ds = dataset.map(tokenize, batched=True)
    
    # SYNCED: Explicitly use DataCollatorWithPadding to prevent sequence length errors
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
    )
    
    print(f"-> Generating predictions for {os.path.basename(model_path)}...")
    preds_output = trainer.predict(tokenized_ds)
    y_pred = np.argmax(preds_output.predictions, axis=-1)
    
    # Cleanup memory before returning
    del model, trainer
    clear_memory()
    
    return y_pred, preds_output.label_ids

def normalize_columns(df):
    if 'text' not in df.columns and 'review' in df.columns:
        df = df.rename(columns={'review': 'text'})
    if 'label' not in df.columns and 'sentiment' in df.columns:
        df = df.rename(columns={'sentiment': 'label'})
    df = df.dropna(subset=['text', 'label'])
    df['label'] = df['label'].astype(int)
    return df

def main():
    print("=== STEP 1: EXTRACTING PREDICTIONS FOR STATISTICAL TEST ===")
    test_path = os.path.join(FIRECS_DIR, 'test.csv')
    
    if not os.path.exists(test_path):
        print(f"[ERROR] Test set missing at {test_path}")
        return

    test_df = normalize_columns(pd.read_csv(test_path))
    test_df['label'] = test_df['label'].astype(int)
    test_ds = Dataset.from_pandas(test_df)
    
    # Extract predictions once before the bootstrap loop to save time
    try:
        y_pred_a, y_true = get_predictions(MODEL_A_PATH, test_ds)
        y_pred_b, _ = get_predictions(MODEL_B_PATH, test_ds)
    except Exception as e:
        print(f"[ERROR] Prediction extraction failed: {e}")
        return
    
    print("\n=== STEP 2: EXECUTING BOOTSTRAP HYPOTHESIS TEST (1000 ITERATIONS) ===")
    n_iterations = 1000
    n_size = len(y_true)
    
    scores_a = []
    scores_b = []
    diff_scores = []
    
    # Set numpy seed for reproducible sampling
    np.random.seed(42) 
    
    for i in range(n_iterations):
        # Resample with replacement (Standard Bootstrap Method)
        indices = np.random.randint(0, n_size, size=n_size)
        
        y_true_boot = y_true[indices]
        y_pred_a_boot = y_pred_a[indices]
        y_pred_b_boot = y_pred_b[indices]
        
        # Calculate Macro F1 for this specific sub-sample
        score_a = f1_score(y_true_boot, y_pred_a_boot, average='macro')
        score_b = f1_score(y_true_boot, y_pred_b_boot, average='macro')
        
        scores_a.append(score_a)
        scores_b.append(score_b)
        diff_scores.append(score_b - score_a)
        
        if (i + 1) % 200 == 0:
            print(f"   -> Completed {i + 1} iterations...")

    # Calculate 95% Confidence Intervals (2.5th to 97.5th percentile)
    ci_lower_a, ci_upper_a = np.percentile(scores_a, [2.5, 97.5])
    ci_lower_b, ci_upper_b = np.percentile(scores_b, [2.5, 97.5])
    
    # Calculate one-tailed p-value (probability that B is NOT better than A)
    p_value = np.sum(np.array(diff_scores) <= 0) / n_iterations

    print("\n=== FINAL STATISTICAL VALIDATION ===")
    print(f"Model A (Generic) 95% CI: [{ci_lower_a:.4f}, {ci_upper_a:.4f}]")
    print(f"Model B (DAPT)    95% CI: [{ci_lower_b:.4f}, {ci_upper_b:.4f}]")
    print(f"Observed P-Value: {p_value:.4f}")
    
    is_significant = p_value < 0.05
    if is_significant:
        print("Result: STATISTICALLY SIGNIFICANT. The DAPT model improvement is mathematically valid.")
    else:
        print("Result: NOT SIGNIFICANT. The margin is within the expected noise of the dataset.")

    # Exporting results for Chapter 4 Tables
    results_df = pd.DataFrame([{
        "Model_A_95_CI": f"[{ci_lower_a:.4f}, {ci_upper_a:.4f}]",
        "Model_B_95_CI": f"[{ci_lower_b:.4f}, {ci_upper_b:.4f}]",
        "P_Value": round(p_value, 4),
        "Significant_at_0.05": "Yes" if is_significant else "No",
        "Iterations": n_iterations
    }])
    
    csv_path = os.path.join(REPORTS_DIR, 'bootstrap_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\n[SUCCESS] Statistical validation results saved to {csv_path}")

if __name__ == "__main__":
    main()