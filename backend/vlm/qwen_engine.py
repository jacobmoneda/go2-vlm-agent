"""Optional Qwen2.5-VL image interpretation. Runs on a separate worker; no motion API."""
import threading
from PIL import Image

PROMPT = '''You are a robot vision observer. Report only visible evidence about the requested target.
Return three short lines: Target:, Appearance:, Uncertainty:.
Do not infer distance, safe travel paths, or issue movement instructions.
If the requested target cannot be verified, say so.'''

class QwenObserver:
    def __init__(self, model_id='Qwen/Qwen2.5-VL-3B-Instruct'):
        self.model_id = model_id
        self._lock = threading.Lock()
        self._model = None
        self._processor = None

    def load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        self._processor = AutoProcessor.from_pretrained(self.model_id, min_pixels=256*28*28, max_pixels=512*28*28)
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(self.model_id, torch_dtype='auto', device_map='auto').eval()

    def describe(self, rgb_image: Image.Image, target: str) -> str:
        import torch
        with self._lock:
            self.load()
            messages = [{'role':'user','content':[{'type':'image'}, {'type':'text','text':f'{PROMPT}\nRequested target: {target}'}]}]
            text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self._processor(text=[text], images=[rgb_image.convert('RGB')], return_tensors='pt')
            inputs = inputs.to(next(self._model.parameters()).device)
            with torch.inference_mode():
                ids = self._model.generate(**inputs, max_new_tokens=90, do_sample=False)
            trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, ids)]
            return self._processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
