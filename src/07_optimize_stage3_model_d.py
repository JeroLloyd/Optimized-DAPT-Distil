import os
import sys
import shutil
import importlib

# --- SUPER-AGGRESSIVE MONKEY PATCH FOR TRANSFORMERS ---
# We must patch the internal structure of transformers BEFORE it is fully used.
import transformers
import transformers.models.auto

# 1. Define the Mock Class
class MockAutoModelForVision2Seq:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        raise NotImplementedError("This is a mock class for compatibility.")

# 2. Inject into the MAIN transformers module
setattr(transformers, "AutoModelForVision2Seq", MockAutoModelForVision2Seq)

# 3. Inject into the AUTO module (Critical for lazy loading resolution)
setattr(transformers.models.auto, "AutoModelForVision2Seq", MockAutoModelForVision2Seq)

# 4. Inject into sys.modules to catch 'from transformers import ...'
if "transformers" in sys.modules:
    sys.modules["transformers"].AutoModelForVision2Seq = MockAutoModelForVision2Seq

# 5. DEEP HACK: Modify the Lazy Module's internal mapping if it exists
# This tricks the lazy loader into thinking the module is already loaded or maps to a valid place
try:
    # If transformers is a lazy module, it has these attributes
    if hasattr(transformers, "_import_structure"):
        transformers._import_structure["models.auto"].append("AutoModelForVision2Seq")
    
    if hasattr(transformers, "_class_to_module"):
        transformers._class_to_module["AutoModelForVision2Seq"] = "models.auto"
except Exception as e:
    pass

# 6. Apply other compatibility patches
if not hasattr(transformers.utils, "is_offline_mode"):
    transformers.utils.is_offline_mode = lambda: False

if not hasattr(transformers.modeling_utils, "get_parameter_dtype"):
    transformers.modeling_utils.get_parameter_dtype = lambda p: p.dtype

if not hasattr(transformers.utils.generic, "_CAN_RECORD_REGISTRY"):
    transformers.utils.generic._CAN_RECORD_REGISTRY = {}

if not hasattr(transformers.utils.generic, "OutputRecorder"):
    class OutputRecorder:
        def __init__(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
    transformers.utils.generic.OutputRecorder = OutputRecorder

# -------------------------------------------------------------------

from transformers import AutoTokenizer
# Import optimum AFTER the patches
from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Input: Model B
INPUT_MODEL = os.path.join(MODELS_DIR, "model_b_dapt")
# Output: Model D
OUTPUT_MODEL = os.path.join(MODELS_DIR, "model_d_onnx")

def main():
    print(f"Project Root: {BASE_DIR}")
    print("--- STAGE 3: ONNX OPTIMIZATION (CPU ONLY) ---")

    if not os.path.exists(INPUT_MODEL):
        print(f"[ERROR] Input model not found: {INPUT_MODEL}")
        return

    if os.path.exists(OUTPUT_MODEL):
        shutil.rmtree(OUTPUT_MODEL)

    try:
        print("Step 1: Exporting to ONNX (Intermediate)...")
        # STRICT CPU ENFORCEMENT via provider
        model = ORTModelForSequenceClassification.from_pretrained(
            INPUT_MODEL, 
            export=True,
            provider="CPUExecutionProvider"
        )
        tokenizer = AutoTokenizer.from_pretrained(INPUT_MODEL)
        
        model.save_pretrained(OUTPUT_MODEL)
        tokenizer.save_pretrained(OUTPUT_MODEL)
        
        print("Step 2: Dynamic Quantization (AVX2)...")
        quantizer = ORTQuantizer.from_pretrained(OUTPUT_MODEL)
        dq_config = AutoQuantizationConfig.avx2(is_static=False, per_channel=False)
        
        quantizer.quantize(
            save_dir=OUTPUT_MODEL,
            quantization_config=dq_config
        )
        
        print("Step 3: Cleaning up artifacts...")
        unquantized_path = os.path.join(OUTPUT_MODEL, "model.onnx")
        quantized_path = os.path.join(OUTPUT_MODEL, "model_quantized.onnx")
        
        if os.path.exists(unquantized_path):
            os.remove(unquantized_path)
            
        if os.path.exists(quantized_path):
            os.rename(quantized_path, unquantized_path)
            
        print(f"SUCCESS: Optimized model saved to {OUTPUT_MODEL}")
        
    except Exception as e:
        print(f"[ERROR] Optimization failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()