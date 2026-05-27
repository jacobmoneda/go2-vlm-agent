#runs qwen on an image file with a pre-determined prompt

from transformers import Qwen2_5_VLForConditionalGeneration
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

import torch


print("Loading Qwen model...")

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-2B-Instruct",
    device_map="auto",
    load_in_4bit=True
)

processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2.5-VL-2B-Instruct"
)

print("Qwen loaded successfully.")


if __name__ == '__main__':

    image_path = ("test_frame_3.jpg")
    prompt = ("Describe the image in two sentences.")

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

    inputs = inputs.to("cuda")

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