from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

import torch
from PIL import Image

MODEL_PATH = "/home/unitree/go2-vlm-agent/models/qwen2-2b"

print("Loading Qwen model...")

model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    device_map="cuda",
    torch_dtype=torch.float16,
)

processor = AutoProcessor.from_pretrained(
    MODEL_PATH,
    min_pixels=256*28*28,
    max_pixels=512*28*28   # limit image resolution to reduce VRAM
)

print("Qwen loaded successfully.")


def run_qwen_with_frame(pil_image, prompt):
    """
    Accepts a PIL Image directly from the camera stream,
    bypassing the need for a file path.
    """

    # Resize image before sending to VLM to reduce memory usage
    img = Image.open(pil_image).convert("RGB")
    img = img.resize((640, 480))
    img.save("/tmp/resized_input.jpg")

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "/tmp/resized_input.jpg", # Use the resized image path for VLM input
                },
                {
                    "type": "text",
                    "text": prompt
                },
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    )

    inputs = inputs.to("cuda")

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    return output_text[0]