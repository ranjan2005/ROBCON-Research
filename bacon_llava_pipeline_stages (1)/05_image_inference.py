# ==============================================================================
# STAGE 5: Downloading Test Image & Running BACON Captioning Inference
# ==============================================================================

import requests
import torch
from PIL import Image
from easydict import EasyDict

from BACON.bacon_modules.llava_module.mm_utils import (
    tokenizer_image_token,
    KeywordsStoppingCriteria,
)
from BACON.bacon_modules.llava_module.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from BACON.bacon_modules.llava_module.conversation import (
    conv_templates,
    SeparatorStyle,
)

# 1. Download sample image
url = "https://images.unsplash.com/photo-1558981806-ec527fa84c39"
image_path = "/kaggle/working/test_bicycle.jpg"

response = requests.get(url, timeout=30)
response.raise_for_status()
with open(image_path, "wb") as f:
    f.write(response.content)

image = Image.open(image_path).convert("RGB")
print("Image loaded:", image.size)

# 2. Define T4-safe inference helper (max_new_tokens=256 to prevent OOM)
def llava_inference(qs, input_image, config, tokenizer, model, image_processor, mode="path"):
    assert mode in ["path", "pil", "tensor"]

    if model.config.mm_use_im_start_end:
        if input_image is not None:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
    elif input_image is not None:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    conv = conv_templates[config.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).cuda()

    if input_image is not None:
        if mode == "path":
            input_image = Image.open(input_image)
        image_tensor = image_processor.preprocess(
            input_image, return_tensors="pt"
        )["pixel_values"][0]

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).half().cuda(),
            do_sample=False,
            num_beams=1,
            max_new_tokens=256,
            use_cache=True
        )

    outputs = tokenizer.batch_decode(
        output_ids, skip_special_tokens=True
    )[0].strip()

    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]

    return outputs.strip()

# 3. Configure and run inference
config = EasyDict()
config.conv_mode = "llava_v1"
config.temperature = 0
config.top_p = None
config.num_beams = 1
config.mm_use_im_start_end = model.config.mm_use_im_start_end

prompt = (
    "Please describe this image in detail. "
    "Specifically, please list all important items and relationships in this image."
)

caption = llava_inference(
    qs=prompt,
    input_image=image,
    config=config,
    tokenizer=tokenizer,
    model=model,
    image_processor=image_processor,
    mode="pil"
)

print("\n=== BACON OUTPUT ===")
print(caption)
