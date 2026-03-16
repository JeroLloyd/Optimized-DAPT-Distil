import os
import pandas as pd
import numpy as np
from datasets import Dataset
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer

# --- PATH CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(SCRIPT_DIR)
BASE_DIR = os.path.dirname(SRC_DIR)
FIRECS_DIR = os.path.join(BASE_DIR, 'data', '03_processed', 'FiReCS_Final')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

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
    print("=== EXTRACTING PREDICTIONS FOR STATISTICAL TEST ===")
    test_df = pd.read_csv(os.path.join(FIRECS_DIR, 'test.csv'))
    test_df['label'] = test_df['label'].astype(int)
    test_ds = Dataset.from_pandas(test_df)
    
    print("Loading Model A (Generic Baseline)...")
    y_pred_a, y_true = get_predictions(MODEL_A_PATH, test_ds)
    
    print("Loading Model B (DAPT Baseline)...")
    y_pred_b, _ = get_predictions(MODEL_B_PATH, test_ds)
    
    print("\n=== EXECUTING BOOTSTRAP HYPOTHESIS TEST (1000 ITERATIONS) ===")
    n_iterations = 1000
    n_size = len(y_true)
    
    scores_a = []
    scores_b = []
    diff_scores = []
    
    np.random.seed(42) # Fixed seed for reproducibility
    
    for i in range(n_iterations):
        # Sample with replacement
        indices = np.random.randint(0, n_size, size=n_size)
        
        y_true_boot = y_true[indices]
        y_pred_a_boot = y_pred_a[indices]
        y_pred_b_boot = y_pred_b[indices]
        
        score_a = f1_score(y_true_boot, y_pred_a_boot, average='macro')
        score_b = f1_score(y_true_boot, y_pred_b_boot, average='macro')
        
        scores_a.append(score_a)
        scores_b.append(score_b)
        diff_scores.append(score_b - score_a)
        
        if (i + 1) % 200 == 0:
            print(f"Completed {i + 1} bootstrap iterations...")

    # Calculate 95% Confidence Intervals
    ci_lower_a, ci_upper_a = np.percentile(scores_a, [2.5, 97.5])
    ci_lower_b, ci_upper_b = np.percentile(scores_b, [2.5, 97.5])
    
    # Calculate p-value (probability that Model A is >= Model B)
    # If p < 0.05, the improvement of Model B is statistically significant.
    p_value = np.sum(np.array(diff_scores) <= 0) / n_iterations

    print("\n=== FINAL STATISTICAL VALIDATION ===")
    print(f"Model A (Generic) 95% CI: [{ci_lower_a:.4f}, {ci_upper_a:.4f}]")
    print(f"Model B (DAPT)    95% CI: [{ci_lower_b:.4f}, {ci_upper_b:.4f}]")
    print(f"P-Value: {p_value:.4f}")
    if p_value < 0.05:
        print("Result: SIGNIFICANT. The DAPT model is mathematically superior.")
    else:
        print("Result: NOT SIGNIFICANT. The improvement could be random noise.")

if __name__ == "__main__":
    main()