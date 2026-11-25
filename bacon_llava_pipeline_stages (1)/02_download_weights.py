# ==============================================================================
# STAGE 2: Downloading Base LLaVA, BACON Captioner & CLIP Vision Tower Weights
# ==============================================================================

import os
from huggingface_hub import snapshot_download

# 1. Download base LLaVA-v1.5-13B model
llava_path = snapshot_download(repo_id="liuhaotian/llava-v1.5-13b")
print("LLaVA downloaded to:", llava_path)

# 2. Download BACON captioner adapter/checkpoint weights
captioner_path = snapshot_download(
    repo_id="ztyang196/bacon-captioner",
    local_dir="/kaggle/working/BACON/ckpt/captioner"
)
print("BACON captioner downloaded to:", captioner_path)

# 3. Download CLIP vision encoder (clip-vit-large-patch14-336)
clip_path = snapshot_download(
    repo_id="openai/clip-vit-large-patch14-336",
    local_dir="/kaggle/working/BACON/ckpt/clip-vit-large-patch14-336"
)
print("CLIP vision tower downloaded to:", clip_path)
