# runs qwen on an image file with a pre-determined prompt

from transformers import Qwen2VLForConditionalGeneration
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from backend.shared_state import shared_state

import torch

#MODEL_PATH = "/home/unitree/go2-vlm-agent/models/qwen2-2b"
MODEL_PATH = "/Users/jmone/models/qwen2-2b"

print("Loading Qwen model...")

model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    device_map="cpu",
    torch_dtype=torch.float16,
)

processor = AutoProcessor.from_pretrained(MODEL_PATH)

print("Qwen loaded successfully.")


if __name__ == '__main__':

    image_path = ("backend/vlm/demo.jpeg")
    #prompt = ("Describe the image in two sentences.")
    prompt = shared_state.latest_prompt

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
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

    inputs = inputs.to("cpu")

    with torch.no_grad():

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=32,
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

    print(output_text[0])
