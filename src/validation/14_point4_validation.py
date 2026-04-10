"""
Validation and Multi-Seed Variance Testing Script.

This module validates the robustness of fine-tuned models across multiple random seeds.
It evaluates Model A and Model B to generate comparative confusion matrices. 
It then performs an automated multi-seed variance test across all models (A, B, C, and 
the quantized Model D) to calculate the mean and standard deviation of accuracy and 
macro F1 scores. Progress is tracked incrementally via a checkpointing system.
"""

import os
import sys
import shutil
import gc

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, 
    TrainingArguments, Trainer, set_seed, EarlyStoppingCallback,
    DataCollatorWithPadding
)

# ==============================================================================
# SUPER-AGGRESSIVE MONKEY PATCH FOR OPTIMUM/ONNX COMPATIBILITY
# ==============================================================================
import transformers
import transformers.models.auto

class MockAutoModelForVision2Seq:
    """Mock class injected into transformers to bypass missing dependencies."""
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        raise NotImplementedError("This is a mock class for compatibility.")

# Apply the mock class to the transformers module
setattr(transformers, "AutoModelForVision2Seq", MockAutoModelForVision2Seq)
setattr(transformers.models.auto, "AutoModelForVision2Seq", MockAutoModelForVision2Seq)

if "transformers" in sys.modules:
    sys.modules["transformers"].AutoModelForVision2Seq = MockAutoModelForVision2Seq

from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig


# ==============================================================================
# DETERMINISTIC SEED LOCK
# ==============================================================================
def lock_environmental_seeds(seed=42):
    """
    Secures all random number generators to ensure reproducible training runs.

    Args:
        seed (int): The numerical seed value applied across all libraries.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    set_seed(seed)
    
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

lock_environmental_seeds(42)


# ==============================================================================
# PATH CONFIGURATION
# ==============================================================================
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
MODEL_C_PATH = os.path.join(MODELS_DIR, "model_c_xlmr")
DAPT_BASE_PATH = os.path.join(MODELS_DIR, 'stage1_dapt_distilmbert') 
BASE_TOKENIZER = "distilbert-base-multilingual-cased"
TRACKING_FILE = os.path.join(METRICS_DIR, 'multi_seed_tracking.csv')


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def clear_memory():
    """
    Forces garbage collection and clears the CUDA memory cache.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_optimizer_grouped_parameters(model, base_lr, weight_decay=0.01):
    """
    Assigns decaying learning rates to model layers. 
    Supports both DistilBERT and XLM-R architectures.

    Args:
        model (PreTrainedModel): The Hugging Face model instance.
        base_lr (float): The starting learning rate for the classifier head.
        weight_decay (float): The weight decay applied to the parameters.

    Returns:
        list: A list of parameter groups formatted for the PyTorch optimizer.
    """
    decay_factor = 0.95
    is_distilbert = hasattr(model, "distilbert")
    
    if is_distilbert:
        layers = model.distilbert.transformer.layer
        embeddings = model.distilbert.embeddings
    else:
        layers = model.roberta.encoder.layer
        embeddings = model.roberta.embeddings
        
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


def get_predictions(model_path, dataset, tokenizer_name=BASE_TOKENIZER):
    """
    Evaluates a model and extracts predictions and true labels.

    Args:
        model_path (str): The directory containing the model.
        dataset (Dataset): The Hugging Face dataset for evaluation.
        tokenizer_name (str): The identifier for the required tokenizer.

    Returns:
        tuple: A tuple containing an array of predicted labels and an array of true labels.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
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


def evaluate_onnx_model(source_model_path, tokenizer_name, test_df, seed):
    """
    Dynamically exports a PyTorch model to ONNX, quantizes it, and evaluates performance.

    Args:
        source_model_path (str): The directory containing the source PyTorch model.
        tokenizer_name (str): The identifier for the tokenizer.
        test_df (pd.DataFrame): The test dataset dataframe.
        seed (int): The current random seed (used for temporary directory naming).

    Returns:
        tuple: Accuracy and macro F1 score of the quantized ONNX model.
    """
    temp_onnx_dir = f"./temp_onnx_seed_{seed}"
    
    # 1. Export to ONNX
    model = ORTModelForSequenceClassification.from_pretrained(
        source_model_path, export=True, provider="CPUExecutionProvider"
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    model.save_pretrained(temp_onnx_dir)
    tokenizer.save_pretrained(temp_onnx_dir)
    
    # 2. Apply Dynamic INT8 Quantization
    quantizer = ORTQuantizer.from_pretrained(temp_onnx_dir)
    dq_config = AutoQuantizationConfig.avx2(is_static=False, per_channel=False)
    quantizer.quantize(save_dir=temp_onnx_dir, quantization_config=dq_config)
    
    # Clean up standard model and keep quantized version
    unquantized_path = os.path.join(temp_onnx_dir, "model.onnx")
    quantized_path = os.path.join(temp_onnx_dir, "model_quantized.onnx")
    
    if os.path.exists(unquantized_path): 
        os.remove(unquantized_path)
    if os.path.exists(quantized_path): 
        os.rename(quantized_path, unquantized_path)
    
    # 3. Evaluate ONNX Model
    ort_model = ORTModelForSequenceClassification.from_pretrained(temp_onnx_dir, provider="CPUExecutionProvider")
    preds = []
    labels = test_df['label'].tolist()
    texts = test_df['text'].tolist()
    
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding='max_length', max_length=128)
        inputs.pop("token_type_ids", None)
        
        outputs = ort_model(**inputs)
        pred = np.argmax(outputs.logits, axis=-1)[0]
        preds.append(pred)
        
    shutil.rmtree(temp_onnx_dir, ignore_errors=True)
    return accuracy_score(labels, preds), f1_score(labels, preds, average='macro')


# ==============================================================================
# CHECKPOINT HANDLING
# ==============================================================================
def load_checkpoints():
    """Loads existing multi-seed evaluation results from the tracking CSV."""
    if os.path.exists(TRACKING_FILE):
        return pd.read_csv(TRACKING_FILE).to_dict('records')
    return []

def save_checkpoint(data):
    """Saves multi-seed evaluation results to the tracking CSV."""
    df = pd.DataFrame(data)
    df.to_csv(TRACKING_FILE, index=False)

def check_if_completed(data, model_name, seed):
    """Checks if a specific model and seed combination has already been evaluated."""
    for entry in data:
        if entry['Model'] == model_name and str(entry['Seed']) == str(seed):
            return entry
    return None


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def main():
    """
    Executes the validation pipeline.
    
    This includes generating comparative confusion matrices for Model A and Model B,
    followed by an automated multi-seed training and evaluation loop to assess 
    variance across all models.
    """
    print("=== PART 1: GENERATING COMPARATIVE CONFUSION MATRIX ===")
    
    test_df = pd.read_csv(os.path.join(FIRECS_DIR, 'test.csv'))
    test_df['label'] = test_df['label'].astype(int)
    test_ds_raw = Dataset.from_pandas(test_df)
    
    # Extract predictions for standard vs domain-adapted models
    y_pred_a, y_true = get_predictions(MODEL_A_PATH, test_ds_raw, "distilbert-base-multilingual-cased")
    y_pred_b, _ = get_predictions(MODEL_B_PATH, test_ds_raw, "distilbert-base-multilingual-cased")
    
    labels = ['Negative', 'Neutral', 'Positive']
    cm_a = confusion_matrix(y_true, y_pred_a)
    cm_b = confusion_matrix(y_true, y_pred_b)

    CM_DIR = os.path.join(FIG_DIR, '07_cm_comparison')
    os.makedirs(CM_DIR, exist_ok=True)
    
    # Save raw confusion matrix data
    pd.DataFrame(cm_a, index=labels, columns=[f"Pred_{l}" for l in labels]).to_csv(os.path.join(CM_DIR, 'cm_comparison_model_a.csv'))
    pd.DataFrame(cm_b, index=labels, columns=[f"Pred_{l}" for l in labels]).to_csv(os.path.join(CM_DIR, 'cm_comparison_model_b.csv'))
    
    # Generate and save the comparative heatmap figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    sns.heatmap(cm_a, annot=True, fmt='d', cmap='Blues', ax=axes[0], xticklabels=labels, yticklabels=labels)
    axes[0].set_title('Generic DistilmBERT (Model A)', fontweight='bold', pad=15)
    
    sns.heatmap(cm_b, annot=True, fmt='d', cmap='Greens', ax=axes[1], xticklabels=labels, yticklabels=labels)
    axes[1].set_title('DAPT-DistilmBERT (Model B)', fontweight='bold', pad=15)
    
    plt.savefig(os.path.join(CM_DIR, 'cm_comparison.png'), dpi=300)
    plt.close()

    # --------------------------------------------------------------------------
    # MULTI-SEED VARIANCE TEST
    # --------------------------------------------------------------------------
    print("\n=== PART 2: AUTOMATED MULTI-SEED VARIANCE TEST (A, B, C, D) ===")
    
    best_lrs = {
        "Model A": 1e-5,
        "Model B": 1e-5,
        "Model C": 1e-5
    }
    
    # Extract optimal learning rates from Stage 2 grid search results
    summary_path = os.path.join(METRICS_DIR, "finetuning_grid_search_results.csv")
    if os.path.exists(summary_path):
        results_df = pd.read_csv(summary_path)
        for model_prefix in best_lrs.keys():
            model_data = results_df[results_df['Model_Name'].str.contains(model_prefix)]
            if not model_data.empty:
                best_lrs[model_prefix] = model_data.loc[model_data['Macro_F1'].idxmax(), 'Learning_Rate']

    # Load full pipeline datasets
    train_df = pd.read_csv(os.path.join(FIRECS_DIR, 'train.csv')).rename(columns={'review':'text'})
    val_df = pd.read_csv(os.path.join(FIRECS_DIR, 'val.csv')).rename(columns={'review':'text'})
    test_df = pd.read_csv(os.path.join(FIRECS_DIR, 'test.csv')).rename(columns={'review':'text'})
    
    train_df['label'], val_df['label'], test_df['label'] = train_df['label'].astype(int), val_df['label'].astype(int), test_df['label'].astype(int)

    configs = [
        {"name": "Model A", "base": "distilbert-base-multilingual-cased", "tok": "distilbert-base-multilingual-cased", "grad_acc": 1, "saved": MODEL_A_PATH},
        {"name": "Model B", "base": DAPT_BASE_PATH, "tok": "distilbert-base-multilingual-cased", "grad_acc": 1, "saved": MODEL_B_PATH},
        {"name": "Model C", "base": "xlm-roberta-base", "tok": "xlm-roberta-base", "grad_acc": 2, "saved": MODEL_C_PATH}
    ]

    seeds = [42, 123, 777]
    checkpoint_data = load_checkpoints()
    
    def compute_metrics_fn(eval_pred):
        preds = np.argmax(eval_pred.predictions, axis=-1)
        return {
            "accuracy": accuracy_score(eval_pred.label_ids, preds),
            "macro_f1": f1_score(eval_pred.label_ids, preds, average='macro')
        }

    # Iterate through models and seeds to evaluate variance
    for config in configs:
        model_name = config["name"]
        lr = best_lrs[model_name]
        print(f"\nEvaluating {model_name} (Best LR: {lr}) across seeds...")
        
        tokenizer = AutoTokenizer.from_pretrained(config["tok"])
        def tokenize_fn(x): return tokenizer(x["text"], truncation=True, max_length=128)
        
        train_ds = Dataset.from_pandas(train_df).map(tokenize_fn, batched=True)
        val_ds = Dataset.from_pandas(val_df).map(tokenize_fn, batched=True)
        test_ds = Dataset.from_pandas(test_df).map(tokenize_fn, batched=True)

        for seed in seeds:
            print(f"  -> Processing Seed: {seed}")
            
            # Check if this seed is already done to save compute time
            existing_result = check_if_completed(checkpoint_data, model_name, seed)
            if existing_result:
                print(f"     [SKIP] Using cached results: Acc {existing_result['Accuracy']}, F1 {existing_result['Macro_F1']}")
                continue

            lock_environmental_seeds(seed)
            temp_output_dir = f"./temp_{model_name.replace(' ', '_')}_seed_{seed}"
            
            # Use pre-trained weights for the baseline seed (42) if available
            if seed == 42 and os.path.exists(config["saved"]):
                model = AutoModelForSequenceClassification.from_pretrained(config["saved"])
                trainer = Trainer(
                    model=model, tokenizer=tokenizer,
                    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
                    compute_metrics=compute_metrics_fn
                )
                eval_res = trainer.evaluate(eval_dataset=test_ds)
                current_model_path = config["saved"]
            else:
                # Train a new instance from scratch for alternate seeds
                model = AutoModelForSequenceClassification.from_pretrained(config["base"], num_labels=3)
                grouped_params = get_optimizer_grouped_parameters(model, lr, weight_decay=0.01)
                optimizer = torch.optim.AdamW(grouped_params)
                
                args = TrainingArguments(
                    output_dir=temp_output_dir, num_train_epochs=15, 
                    learning_rate=lr, per_device_train_batch_size=16,
                    gradient_accumulation_steps=config["grad_acc"],
                    evaluation_strategy="epoch", save_strategy="epoch", 
                    load_best_model_at_end=True, metric_for_best_model="macro_f1", 
                    greater_is_better=True, report_to="none",
                    label_smoothing_factor=0.1, warmup_ratio=0.10, weight_decay=0.01
                )
                
                trainer = Trainer(
                    model=model, args=args, optimizers=(optimizer, None),
                    train_dataset=train_ds, eval_dataset=val_ds, tokenizer=tokenizer,
                    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
                    compute_metrics=compute_metrics_fn,
                    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
                )
                
                trainer.train()
                eval_res = trainer.evaluate(eval_dataset=test_ds)
                trainer.save_model(temp_output_dir)
                current_model_path = temp_output_dir

            acc = eval_res['eval_accuracy']
            f1 = eval_res['eval_macro_f1']
            
            # Save progress immediately
            checkpoint_data.append({
                "Model": model_name, "Seed": seed, 
                "Accuracy": round(acc, 4), "Macro_F1": round(f1, 4)
            })
            save_checkpoint(checkpoint_data)

            # Extrapolate Model D by dynamically quantizing the Model B instance
            if model_name == "Model B":
                existing_d = check_if_completed(checkpoint_data, "Model D", seed)
                if not existing_d:
                    print(f"     -> Quantizing Model B to ONNX (Model D) for Seed: {seed}")
                    d_acc, d_f1 = evaluate_onnx_model(current_model_path, config["tok"], test_df, seed)
                    checkpoint_data.append({
                        "Model": "Model D", "Seed": seed, 
                        "Accuracy": round(d_acc, 4), "Macro_F1": round(d_f1, 4)
                    })
                    save_checkpoint(checkpoint_data)
                else:
                    print(f"     [SKIP] Model D ONNX using cached results for seed {seed}.")

            # Clean up memory and temporary files
            del model, trainer
            clear_memory()
            if seed != 42 and os.path.exists(temp_output_dir): 
                shutil.rmtree(temp_output_dir, ignore_errors=True)

    # --------------------------------------------------------------------------
    # AGGREGATE FINAL RESULTS
    # --------------------------------------------------------------------------
    print("\n--- Compiling Final Variance Report ---")
    final_output_data = list(checkpoint_data) 
    
    unique_models = pd.DataFrame(checkpoint_data)['Model'].unique()
    
    # Calculate Mean and Standard Deviation for each model
    for mod in unique_models:
        model_rows = [row for row in checkpoint_data if row['Model'] == mod and str(row['Seed']) in ['42', '123', '777']]
        
        if len(model_rows) == 3:
            accs = [r['Accuracy'] for r in model_rows]
            f1s = [r['Macro_F1'] for r in model_rows]
            
            final_output_data.append({
                "Model": mod, "Seed": "Mean", 
                "Accuracy": round(np.mean(accs), 4), 
                "Macro_F1": round(np.mean(f1s), 4)
            })
            final_output_data.append({
                "Model": mod, "Seed": "Standard Deviation", 
                "Accuracy": round(np.std(accs), 4), 
                "Macro_F1": round(np.std(f1s), 4)
            })

    output_path = os.path.join(METRICS_DIR, 'multi_seed_variance_results.csv')
    pd.DataFrame(final_output_data).to_csv(output_path, index=False)
    
    print(f"\n[SUCCESS] Comprehensive Variance Results saved to {output_path}")


if __name__ == "__main__":
    main()