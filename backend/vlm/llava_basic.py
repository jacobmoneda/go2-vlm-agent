from PIL import Image
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

# 1. Load model and processor
model_id = "llava-hf/llava-1.5-7b-hf"
model = LlavaForConditionalGeneration.from_pretrained(
    model_id, torch_dtype=torch.float16, device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_id)

# 2. Prepare image and prompt
image = Image.open("path_to_your_local_image.jpg")
prompt = "<image>\nUSER: What is in this image?\nASSISTANT:"

# 3. Process inputs
inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)

# 4. Generate and decode
generate_ids = model.generate(**inputs, max_new_tokens=50)
print(processor.batch_decode(generate_ids, skip_special_tokens=True)[0])