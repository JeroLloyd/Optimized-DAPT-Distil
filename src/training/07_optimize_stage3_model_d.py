import os
import sys
import shutil

# --- SUPER-AGGRESSIVE MONKEY PATCH FOR TRANSFORMERS ---
import transformers
import transformers.models.auto

class MockAutoModelForVision2Seq:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        raise NotImplementedError("This is a mock class for compatibility.")

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
from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(SCRIPT_DIR)
BASE_DIR = os.path.dirname(SRC_DIR)
MODELS_DIR = os.path.join(BASE_DIR, 'models')

INPUT_MODEL = os.path.join(MODELS_DIR, "model_b_dapt")
OUTPUT_MODEL = os.path.join(MODELS_DIR, "model_d_onnx")

def main():
    print(f"Project Root: {BASE_DIR}")
    print("--- STAGE 3: ONNX INT8 DYNAMIC QUANTIZATION ---")

    if not os.path.exists(INPUT_MODEL):
        print(f"[ERROR] Input model not found: {INPUT_MODEL}")
        return

    if os.path.exists(OUTPUT_MODEL):
        shutil.rmtree(OUTPUT_MODEL)

    try:
        print("Step 1: Exporting to ONNX (Intermediate)...")
        model = ORTModelForSequenceClassification.from_pretrained(
            INPUT_MODEL, 
            export=True,
            provider="CPUExecutionProvider"
        )
        tokenizer = AutoTokenizer.from_pretrained(INPUT_MODEL)
        
        model.save_pretrained(OUTPUT_MODEL)
        tokenizer.save_pretrained(OUTPUT_MODEL)
        
        print("Step 2: Applying INT8 Dynamic Quantization...")
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
            
        print(f"SUCCESS: Quantized model saved to {OUTPUT_MODEL}")
       
        
    except Exception as e:
        print(f"[ERROR] Optimization failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()