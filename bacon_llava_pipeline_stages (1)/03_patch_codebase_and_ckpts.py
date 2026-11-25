# ==============================================================================
# STAGE 3: Patching BACON Codebase & Checkpoints for Transformers Compatibility
# ==============================================================================

import os
import json
import shutil
import torch
from pathlib import Path

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# ------------------------------------------------------------------------------
# Patch 1: Fix LlavaConfig model_type and AutoConfig registration in llava_llama.py
# ------------------------------------------------------------------------------
llava_llama_path = Path("/kaggle/working/BACON/bacon_modules/llava_module/model/language_model/llava_llama.py")
text = llava_llama_path.read_text()

text = text.replace(
    'class LlavaConfig(LlamaConfig):\n    model_type = "llava"',
    'class LlavaConfig(LlamaConfig):\n    model_type = "llava_llama"'
)
text = text.replace(
    'AutoConfig.register("llava", LlavaConfig)',
    'AutoConfig.register("llava_llama", LlavaConfig, exist_ok=True)'
)
text = text.replace(
    'AutoConfig.register("llava", LlavaConfig, exist_ok=True)',
    'AutoConfig.register("llava_llama", LlavaConfig, exist_ok=True)'
)
llava_llama_path.write_text(text)
print("✅ Patched llava_llama.py")

# ------------------------------------------------------------------------------
# Patch 2: Remove unused MPT/Mistral imports in model/__init__.py
# ------------------------------------------------------------------------------
init_path = Path("/kaggle/working/BACON/bacon_modules/llava_module/model/__init__.py")
init_text = init_path.read_text()

old_imports = """from .language_model.llava_llama import LlavaLlamaForCausalLM, LlavaConfig
from .language_model.llava_mpt import LlavaMptForCausalLM, LlavaMptConfig
"""
new_imports = """from .language_model.llava_llama import LlavaLlamaForCausalLM, LlavaConfig
"""
if old_imports in init_text:
    init_text = init_text.replace(old_imports, new_imports)

old_indented = """    from .language_model.llava_llama import LlavaLlamaForCausalLM, LlavaConfig
    from .language_model.llava_mpt import LlavaMptForCausalLM, LlavaMptConfig
    # from .language_model.llava_mistral import LlavaMistralForCausalLM, LlavaMistralConfig
"""
new_indented = """    from .language_model.llava_llama import LlavaLlamaForCausalLM, LlavaConfig
"""
if old_indented in init_text:
    init_text = init_text.replace(old_indented, new_indented)

init_path.write_text(init_text)
print("✅ Patched model/__init__.py")

# ------------------------------------------------------------------------------
# Patch 3: Inject custom LLaVASentencePieceTokenizer & config loading into builder.py
# ------------------------------------------------------------------------------
builder_path = Path("/kaggle/working/BACON/bacon_modules/llava_module/model/builder.py")
shutil.copy2(builder_path, builder_path.with_suffix(".py.backup"))
b_text = builder_path.read_text()

if "import json" not in b_text:
    b_text = b_text.replace("import os\n", "import os\nimport json\n", 1)
if "import sentencepiece as spm" not in b_text:
    b_text = b_text.replace("import torch", "import torch\nimport sentencepiece as spm", 1)
if "from transformers import PreTrainedTokenizer" not in b_text:
    b_text = b_text.replace(
        "from transformers import",
        "from transformers import PreTrainedTokenizer\nfrom transformers import",
        1
    )

tokenizer_class_code = r'''
class LLaVASentencePieceTokenizer(PreTrainedTokenizer):
    """SentencePiece tokenizer compatible with the original LLaVA v1.5 tokenizer.model."""

    vocab_files_names = {"vocab_file": "tokenizer.model"}
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<unk>",
        model_max_length=2048,
        add_bos_token=True,
        add_eos_token=False,
        **kwargs,
    ):
        self.vocab_file = vocab_file
        self.sp_model = spm.SentencePieceProcessor()

        if not self.sp_model.Load(vocab_file):
            raise ValueError(f"Could not load SentencePiece model: {vocab_file}")

        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token

        super().__init__(
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            model_max_length=model_max_length,
            **kwargs,
        )

    @property
    def vocab_size(self):
        return self.sp_model.GetPieceSize()

    def get_vocab(self):
        return {
            self.sp_model.IdToPiece(i): i
            for i in range(self.sp_model.GetPieceSize())
        }

    def _tokenize(self, text, **kwargs):
        return self.sp_model.EncodeAsPieces(text)

    def _convert_token_to_id(self, token):
        return self.sp_model.PieceToId(token)

    def _convert_id_to_token(self, index):
        return self.sp_model.IdToPiece(index)

    def convert_tokens_to_string(self, tokens):
        return self.sp_model.DecodePieces(tokens)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        output = []
        if self.add_bos_token:
            output.append(self.bos_token_id)
        output.extend(token_ids_0)
        if token_ids_1 is not None:
            output.append(self.eos_token_id)
            output.extend(token_ids_1)
        if self.add_eos_token:
            output.append(self.eos_token_id)
        return output

    def save_vocabulary(self, save_directory, filename_prefix=None):
        os.makedirs(save_directory, exist_ok=True)
        filename = f"{filename_prefix}-tokenizer.model" if filename_prefix else "tokenizer.model"
        output_file = os.path.join(save_directory, filename)
        with open(self.vocab_file, "rb") as src, open(output_file, "wb") as dst:
            dst.write(src.read())
        return (output_file,)
'''

if "class LLaVASentencePieceTokenizer" not in b_text:
    b_text = b_text.replace("def load_pretrained_model(", tokenizer_class_code + "\n\ndef load_pretrained_model(", 1)

# Replace AutoTokenizer calls with LLaVASentencePieceTokenizer
b_text = b_text.replace(
    'tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)',
    'tokenizer = LLaVASentencePieceTokenizer(vocab_file=os.path.join(model_path, "tokenizer.model"), model_max_length=2048, add_bos_token=True, add_eos_token=False)'
)
b_text = b_text.replace(
    'tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)',
    'tokenizer = LLaVASentencePieceTokenizer(vocab_file=os.path.join(model_base, "tokenizer.model"), model_max_length=2048, add_bos_token=True, add_eos_token=False)'
)

# Fix cfg_pretrained loading in builder.py
b_text = b_text.replace(
    'cfg_pretrained = AutoConfig.from_pretrained(model_path)',
    'cfg_pretrained = LlavaConfig(**json.load(open(os.path.join(model_path, "config.json"))))'
)
builder_path.write_text(b_text)
print("✅ Patched builder.py")

# ------------------------------------------------------------------------------
# Patch 4: Update mm_vision_tower path in BACON config.json
# ------------------------------------------------------------------------------
config_path = "/kaggle/working/BACON/ckpt/captioner/config.json"
with open(config_path, "r") as f:
    cfg = json.load(f)

cfg["mm_vision_tower"] = "/kaggle/working/BACON/ckpt/clip-vit-large-patch14-336"

with open(config_path, "w") as f:
    json.dump(cfg, f, indent=2)
print("✅ Updated mm_vision_tower path in config.json")

# ------------------------------------------------------------------------------
# Patch 5: Extract mm_projector.bin from non_lora_trainables.bin
# ------------------------------------------------------------------------------
src_weights = "/kaggle/working/BACON/ckpt/captioner/non_lora_trainables.bin"
dst_projector = "/kaggle/working/BACON/ckpt/captioner/mm_projector.bin"

weights = torch.load(src_weights, map_location="cpu", weights_only=False)
projector = {
    "mm_projector.0.weight": weights["base_model.model.model.mm_projector.0.weight"],
    "mm_projector.0.bias": weights["base_model.model.model.mm_projector.0.bias"],
    "mm_projector.2.weight": weights["base_model.model.model.mm_projector.2.weight"],
    "mm_projector.2.bias": weights["base_model.model.model.mm_projector.2.bias"],
}
torch.save(projector, dst_projector)
print("✅ Extracted mm_projector.bin")
