# ==============================================================================
# STAGE 1: Environment Setup, Repository Cloning & Dependency Installation
# ==============================================================================

# 1. Install/upgrade HuggingFace transformers
!pip install -U transformers

# 2. Check available disk space
import shutil
total, used, free = shutil.disk_usage("/")
print(f"Total: {total/1024**3:.1f} GB | Used: {used/1024**3:.1f} GB | Free: {free/1024**3:.1f} GB")

# 3. Clone the BACON repository and prepare checkpoint directories
%cd /kaggle/working
!git clone https://github.com/ztyang23/BACON.git

import os
os.makedirs("/kaggle/working/BACON/ckpt/captioner", exist_ok=True)

# 4. Install required packages, CLIP, and BitsAndBytes
!pip install -q easydict==1.13 einops==0.7.0 peft==0.10.0 sentencepiece==0.1.99 unidecode opencv-python
!pip install -q git+https://github.com/openai/CLIP.git
!pip install -q -U bitsandbytes
!pip install -q protobuf==3.20.3

# Optional segmentation/detection dependencies explored in notebook
!pip install -q git+https://github.com/IDEA-Research/GroundingDINO.git
!pip install -q git+https://github.com/facebookresearch/segment-anything.git

# 5. Verify GPU, PyTorch, and library versions
import torch
import transformers
import bitsandbytes
import sentencepiece
import tokenizers
import google.protobuf

print("PyTorch      :", torch.__version__)
print("CUDA         :", torch.cuda.is_available())
print("GPU          :", torch.cuda.get_device_name(0))
print("Transformers :", transformers.__version__)
print("BitsAndBytes :", bitsandbytes.__version__)
print("SentencePiece:", sentencepiece.__version__)
print("Tokenizers   :", tokenizers.__version__)
print("Protobuf     :", google.protobuf.__version__)
