import os
import random
import torch
import numpy as np
import transformers.utils
import transformers.modeling_utils
import transformers.models.auto
import transformers.utils.generic
from transformers import AutoTokenizer

from config import SEED

def set_global_seed():
    """Sets the seed for reproducibility across all computing libraries."""
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(SEED)
    print(f"[INFO] Global Seed locked to {SEED}.")

def apply_transformers_patches():
    """Applies necessary compatibility patches for offline ONNX export."""
    if not hasattr(transformers.utils, 'is_offline_mode'):
        transformers.utils.is_offline_mode = lambda: False

    if not hasattr(transformers.modeling_utils, 'get_parameter_dtype'):
        def _mock_get_parameter_dtype(model):
            try:
                return next(model.parameters()).dtype
            except Exception:
                return torch.float32
        transformers.modeling_utils.get_parameter_dtype = _mock_get_parameter_dtype

    class MockAutoModelForVision2Seq:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            raise NotImplementedError("Mock class for compatibility.")

    setattr(transformers, "AutoModelForVision2Seq", MockAutoModelForVision2Seq)
    setattr(transformers.models.auto, "AutoModelForVision2Seq", MockAutoModelForVision2Seq)

    if not hasattr(transformers.utils.generic, '_CAN_RECORD_REGISTRY'):
        transformers.utils.generic._CAN_RECORD_REGISTRY = {}

    if not hasattr(transformers.utils.generic, 'OutputRecorder'):
        class MockOutputRecorder:
            pass
        transformers.utils.generic.OutputRecorder = MockOutputRecorder

def safe_load_tokenizer(model_path, fallback_name="distilbert-base-multilingual-cased"):
    """Bypasses fast tokenizer corruption issues with a fallback mechanism."""
    try:
        return AutoTokenizer.from_pretrained(model_path)
    except Exception as e:
        print(f"  [WARN] Fast tokenizer failed: {e}. Retrying with use_fast=False...")
        try:
            return AutoTokenizer.from_pretrained(model_path, use_fast=False)
        except Exception:
            print(f"  [WARN] Local tokenizer failed completely. Falling back to base {fallback_name}...")
            return AutoTokenizer.from_pretrained(fallback_name)