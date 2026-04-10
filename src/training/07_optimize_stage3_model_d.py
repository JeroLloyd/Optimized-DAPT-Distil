"""
Stage 3 ONNX Conversion and Quantization Script.

This module converts a fine-tuned Hugging Face transformer model into the ONNX format. 
It applies INT8 dynamic quantization using the Optimum library to reduce model footprint 
and accelerate inference times. It also includes compatibility patches for the 
transformers library.
"""

import os
import sys
import shutil
import time
import traceback

import pandas as pd
import transformers
import transformers.models.auto

# ==============================================================================
# TRANSFORMERS COMPATIBILITY PATCH
# ==============================================================================
# This section applies aggressive monkey patches to the transformers library.
# It ensures compatibility with specific environments missing Vision2Seq models.

class MockAutoModelForVision2Seq:
    """Provides a mock implementation to prevent import errors during initialization."""
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        raise NotImplementedError("This is a mock class for compatibility.")

# Apply the mock class to the transformers module
setattr(transformers, "AutoModelForVision2Seq", MockAutoModelForVision2Seq)
setattr(transformers.models.auto, "AutoModelForVision2Seq", MockAutoModelForVision2Seq)

if "transformers" in sys.modules:
    sys.modules["transformers"].AutoModelForVision2Seq = MockAutoModelForVision2Seq

try:
    if hasattr(transformers, "_import_structure"):
        transformers._import_structure["models.auto"].append("AutoModelForVision2Seq")
    if hasattr(transformers, "_class_to_module"):
        transformers._class_to_module["AutoModelForVision2Seq"] = "models.auto"
except Exception:
    pass

# Bypass offline mode checks
if not hasattr(transformers.utils, "is_offline_mode"):
    transformers.utils.is_offline_mode = lambda: False

# Bypass parameter datatype checks
if not hasattr(transformers.modeling_utils, "get_parameter_dtype"):
    transformers.modeling_utils.get_parameter_dtype = lambda p: p.dtype

if not hasattr(transformers.utils.generic, "_CAN_RECORD_REGISTRY"):
    transformers.utils.generic._CAN_RECORD_REGISTRY = {}

if not hasattr(transformers.utils.generic, "OutputRecorder"):
    class OutputRecorder:
        """Mocks the OutputRecorder context manager for legacy support."""
        def __init__(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
    transformers.utils.generic.OutputRecorder = OutputRecorder


# ==============================================================================
# OPTIMUM IMPORTS AND PATH CONFIGURATION
# ==============================================================================
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(SCRIPT_DIR)
BASE_DIR = os.path.dirname(SRC_DIR)
MODELS_DIR = os.path.join(BASE_DIR, 'models')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports', 'metrics')

os.makedirs(REPORTS_DIR, exist_ok=True)

INPUT_MODEL = os.path.join(MODELS_DIR, "model_b_dapt")
OUTPUT_MODEL = os.path.join(MODELS_DIR, "model_d_onnx")


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def get_dir_size_mb(path):
    """
    Calculates the total size of a given directory in megabytes.

    Args:
        path (str): The target directory path.

    Returns:
        float: The total directory size in MB.
    """
    total_size = 0
    if not os.path.exists(path): 
        return 0
        
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # Ignore symbolic links and intermediate checkpoint files
            if not os.path.islink(fp) and "checkpoint" not in fp:
                total_size += os.path.getsize(fp)
                
    return total_size / (1024 * 1024)


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def main():
    """
    Executes the ONNX conversion and INT8 dynamic quantization pipeline.
    
    This process exports the standard model to ONNX format. It then applies 
    dynamic quantization to the model weights and calculates the final 
    compression metrics for reporting.
    """
    print(f"Project Root: {BASE_DIR}")
    print("--- STAGE 3: ONNX INT8 DYNAMIC QUANTIZATION ---")

    # --------------------------------------------------------------------------
    # 1. VALIDATION
    # --------------------------------------------------------------------------
    if not os.path.exists(INPUT_MODEL):
        print(f"[ERROR] Input model not found: {INPUT_MODEL}")
        return

    # Remove existing output to prevent artifact conflicts
    if os.path.exists(OUTPUT_MODEL):
        shutil.rmtree(OUTPUT_MODEL)

    start_time = time.time()

    try:
        # ----------------------------------------------------------------------
        # 2. ONNX EXPORT
        # ----------------------------------------------------------------------
        print("Step 1: Exporting to ONNX (Intermediate)...")
        model = ORTModelForSequenceClassification.from_pretrained(
            INPUT_MODEL, 
            export=True,
            provider="CPUExecutionProvider"
        )
        tokenizer = AutoTokenizer.from_pretrained(INPUT_MODEL)
        
        model.save_pretrained(OUTPUT_MODEL)
        tokenizer.save_pretrained(OUTPUT_MODEL)
        
        # ----------------------------------------------------------------------
        # 3. INT8 DYNAMIC QUANTIZATION
        # ----------------------------------------------------------------------
        print("Step 2: Applying INT8 Dynamic Quantization...")
        quantizer = ORTQuantizer.from_pretrained(OUTPUT_MODEL)
        dq_config = AutoQuantizationConfig.avx2(is_static=False, per_channel=False)
        
        quantizer.quantize(
            save_dir=OUTPUT_MODEL,
            quantization_config=dq_config
        )
        
        # ----------------------------------------------------------------------
        # 4. ARTIFACT CLEANUP
        # ----------------------------------------------------------------------
        print("Step 3: Cleaning up artifacts...")
        unquantized_path = os.path.join(OUTPUT_MODEL, "model.onnx")
        quantized_path = os.path.join(OUTPUT_MODEL, "model_quantized.onnx")
        
        # Delete the bulky unquantized model
        if os.path.exists(unquantized_path):
            os.remove(unquantized_path)
            
        # Rename the quantized model to the standard target name
        if os.path.exists(quantized_path):
            os.rename(quantized_path, unquantized_path)
            
        end_time = time.time()
        
        # ----------------------------------------------------------------------
        # 5. METRICS CALCULATION AND EXPORT
        # ----------------------------------------------------------------------
        orig_size = get_dir_size_mb(INPUT_MODEL)
        quant_size = get_dir_size_mb(OUTPUT_MODEL)
        reduction_pct = ((orig_size - quant_size) / orig_size) * 100 if orig_size > 0 else 0
        
        print(f"SUCCESS: Quantized model saved to {OUTPUT_MODEL}")
        
        # Format compression metrics
        metrics_df = pd.DataFrame([{
            "Original_Model": "Model B (DAPT-DistilBERT)",
            "Optimized_Model": "Model D (ONNX INT8)",
            "Original_Size_MB": round(orig_size, 2),
            "Quantized_Size_MB": round(quant_size, 2),
            "Size_Reduction_Pct": round(reduction_pct, 2),
            "Processing_Time_s": round(end_time - start_time, 2)
        }])
        
        metrics_path = os.path.join(REPORTS_DIR, "stage3_quantization_metrics.csv")
        metrics_df.to_csv(metrics_path, index=False)
        print(f"SUCCESS: Compression metrics saved to {metrics_path}")
        
    except Exception as e:
        print(f"[ERROR] Optimization failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()