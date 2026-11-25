# ==============================================================================
# STAGE 4: Loading the Patched BACON + LLaVA-v1.5-13B Model
# ==============================================================================

import sys
import importlib
import torch

sys.path.insert(0, "/kaggle/working")
sys.path.append("/kaggle/working/BACON")

import BACON.bacon_modules.llava_module.model.language_model.llava_llama as llava_llama_module
import BACON.bacon_modules.llava_module.model.builder as builder

importlib.reload(llava_llama_module)
builder = importlib.reload(builder)

LLAVA_PATH = "/root/.cache/huggingface/hub/models--liuhaotian--llava-v1.5-13b/snapshots/901a44b9113dea67b976e71f58d4e372cf9de81a"
BACON_PATH = "/kaggle/working/BACON/ckpt/captioner"

print("Loading BACON + LLaVA-v1.5-13B...")
tokenizer, model, image_processor, context_len = builder.load_pretrained_model(
    model_path=BACON_PATH,
    model_base=LLAVA_PATH,
    model_name="llava-v1.5-13b",
    load_4bit=False,
    device_map="auto",
    device="cuda"
)

print("✅ MODEL LOADED SUCCESSFULLY")
print("Tokenizer type :", type(tokenizer))
print("Model type     :", type(model))
print("Context length :", context_len)
print("Image processor:", type(image_processor))
