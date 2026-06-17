import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL_ID = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"

print("[SmolVLM2] Loading model...")

device = "cuda" if torch.cuda.is_available() else "cpu"
print("[SmolVLM2] Device:", device)

processor = AutoProcessor.from_pretrained(MODEL_ID)

model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto" if device == "cuda" else None
)

if device == "cpu":
    model = model.to("cpu")

print("[SmolVLM2] Model loaded successfully.")


def run_smolvlm_with_frame(pil_image, prompt):
    """
    Run SmolVLM2 on one PIL image and return only the newly generated answer.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": prompt}
            ]
        }
    ]

    # Newer transformers API
    try:
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            processor_kwargs={
                "return_tensors": "pt"
            }
        )
    except TypeError:
        # Fallback for older versions
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        )

    inputs = inputs.to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False
        )

    # Keep only the newly generated AI answer
    generated_ids_trimmed = generated_ids[:, inputs["input_ids"].shape[-1]:]

    answer = processor.decode(
        generated_ids_trimmed[0],
        skip_special_tokens=True
    )

    return answer.strip()