from transformers import AutoModelForCausalLM, AutoProcessor
import torch
from PIL import Image

MODEL_PATH = "/home/unitree/models/Phi-3.5-vision-instruct"

_model = None
_processor = None

def _load_model():
    global _model, _processor
    if _model is None:
        print("[Phi] Loading model...")
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            device_map="cuda",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            _attn_implementation="eager"
        )
        _processor = AutoProcessor.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            num_crops=1
        )
        print("[Phi] Model loaded.")
    return _model, _processor

def run_phi_with_frame(pil_image: Image.Image, prompt: str) -> str:
    model, processor = _load_model()
    """
    Accepts a PIL Image directly from the camera stream,
    bypassing the need for a file path.
    """
    messages = [
        {
            "role": "system",
            "content": "You are a robot vision system. Respond ONLY with a JSON object. No explanation, no markdown, no extra text."
        },
        {
            "role": "user",
            "content": f"<|image_1|>\n{prompt}"
        }
    ]

    text = processor.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = processor(
        text=text,
        images=[pil_image],
        return_tensors="pt"
    ).to("cuda")

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            eos_token_id=processor.tokenizer.eos_token_id
        )

    generated_ids_trimmed = generated_ids[:, inputs["input_ids"].shape[1]:]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    return output_text[0]