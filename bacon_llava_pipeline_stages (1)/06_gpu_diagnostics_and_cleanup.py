# ==============================================================================
# STAGE 6: GPU Memory Diagnostics, Device Inspection & VRAM Cleanup
# ==============================================================================

import gc
import torch
from collections import Counter

# 1. Inspect parameter distribution across devices (if model is still in memory)
if "model" in globals():
    device_counts = Counter()
    for name, param in model.named_parameters():
        device_counts[str(param.device)] += param.numel()

    for device, numel in device_counts.items():
        print(f"{device}: {numel / 1e9:.2f} billion parameters")

    print("Model device map:", getattr(model, "hf_device_map", "No hf_device_map found"))

# 2. Inspect active CUDA tensors
gc.collect()
cuda_tensors = []
for obj in gc.get_objects():
    try:
        if torch.is_tensor(obj) and obj.is_cuda:
            size = obj.numel() * obj.element_size()
            cuda_tensors.append((size, obj))
    except Exception:
        pass

cuda_tensors.sort(reverse=True, key=lambda x: x[0])
print(f"CUDA tensors found: {len(cuda_tensors)}")
for i, (size, tensor) in enumerate(cuda_tensors[:10]):
    print(f"{i}: {size/1024**3:.3f} GB | shape={tuple(tensor.shape)} | dtype={tensor.dtype} | device={tensor.device}")

# 3. Delete model objects and flush VRAM
_names = [
    "model", "tokenizer", "image_processor", "vision_tower", "config",
    "cuda_tensors", "tensor", "weight", "refs", "output_ids", "input_ids",
    "image_tensor", "stopping_criteria", "conv", "prompt", "qs",
]

for name in _names:
    if name in globals():
        try:
            del globals()[name]
        except Exception:
            pass

gc.collect()
torch.cuda.empty_cache()
torch.cuda.ipc_collect()

free_bytes, total_bytes = torch.cuda.mem_get_info()
print(f"GPU free      : {free_bytes / 1024**3:.2f} GB")
print(f"GPU total     : {total_bytes / 1024**3:.2f} GB")
print(f"GPU allocated : {torch.cuda.memory_allocated()/1024**3:.2f} GB")
print(f"GPU reserved  : {torch.cuda.memory_reserved()/1024**3:.2f} GB")
