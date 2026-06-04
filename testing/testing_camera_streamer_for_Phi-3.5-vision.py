import sys
import time
import queue
import os
import cv2
import torch
import numpy as np
from PIL import Image

# =========================
# Fix PIL is_directory error
# =========================
import PIL._util

if not hasattr(PIL._util, "is_directory"):
    PIL._util.is_directory = lambda path: os.path.isdir(path)

from transformers import AutoModelForCausalLM, AutoProcessor

from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_go.msg.dds_ import Go2FrontVideoData_


# =========================
# Settings
# =========================

DEFAULT_INTERFACE = "eth0"

MODEL_ID = "microsoft/Phi-3.5-vision-instruct"

IMAGE_SIZE = (224, 224)

SHOW_CAMERA_WINDOW = True

SAVE_PNG = True
PNG_FILENAME = "latest_go2_phi_frame.png"


# =========================
# Unitree Go2 Camera Stream
# =========================

class Go2CameraStreamer:
    def __init__(self, interface: str):
        self.interface = interface
        self.frame_queue = queue.Queue(maxsize=2)
        self.is_running = False
        self.sub = None

    def _video_callback(self, msg: Go2FrontVideoData_):
        if msg.video720p and len(msg.video720p) > 0:
            np_arr = np.frombuffer(bytes(msg.video720p), dtype=np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is not None:
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass

                self.frame_queue.put_nowait(frame)

    def start(self):
        ChannelFactoryInitialize(0, self.interface)
        self.sub = ChannelSubscriber("/frontvideostream", Go2FrontVideoData_)
        self.sub.Init(self._video_callback, 10)
        self.is_running = True
        print(f"[SDK] Subscribed to Go2 camera stream on {self.interface}")

    def get_latest_frame(self, timeout=None):
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None


# =========================
# Load Phi-3.5-Vision
# =========================

def load_phi_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[AI] Device:", device)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        _attn_implementation="eager"
    )

    if device == "cpu":
        model = model.to("cpu")

    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        num_crops=1
    )

    print("[AI] Phi-3.5-Vision loaded successfully")

    return model, processor, device


# =========================
# Robot Vision Prompt
# =========================

ROBOT_PROMPT = """
You are the vision system of a robot dog.

Look at the image and return only these three lines:

Type: choose one from human, animal, or object.
Hand gesture: describe only the visible hand/finger gesture.
Location: choose indoor or outdoor.

Do NOT describe clothing, glasses, face details, room details, ceiling, lights, or background objects.
Do NOT describe the person's intention or activity, such as taking a selfie.
Only describe the visible hand gesture and movement.

Focus on these gestures:
- five fingers spread up
- index finger pointing up
- finger pointing forward
- peace sign
- three fingers raised
- fist raised
- thumbs up
- waving hand
- no clear hand gesture

Use this exact format:
Type:
Hand gesture:
Location:
"""


# =========================
# Convert Go2 Frame to PIL
# =========================

def go2_frame_to_pil(frame_bgr):
    """
    Go2/OpenCV frame is BGR.
    Phi/PIL image needs RGB.
    """

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb).convert("RGB")
    image = image.resize(IMAGE_SIZE)

    if SAVE_PNG:
        image.save(PNG_FILENAME, format="PNG")

    return image


# =========================
# Phi Vision Function
# =========================

def run_phi_vision(image, model, processor, device):
    messages = [
        {
            "role": "user",
            "content": "<|image_1|>\n" + ROBOT_PROMPT
        }
    ]

    prompt = processor.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = processor(
        prompt,
        [image],
        return_tensors="pt"
    )

    input_device = "cuda:0" if device == "cuda" else "cpu"
    inputs = {k: v.to(input_device) for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False,
            eos_token_id=processor.tokenizer.eos_token_id
        )

    # Keep only the newly generated AI answer
    generated_ids_trimmed = generated_ids[:, inputs["input_ids"].shape[1]:]

    output = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    return output[0].strip()


# =========================
# Main Loop
# =========================

def main():
    if len(sys.argv) >= 2:
        interface = sys.argv[1]
    else:
        interface = DEFAULT_INTERFACE

    print("[MAIN] Using network interface:", interface)

    model, processor, device = load_phi_model()

    camera = Go2CameraStreamer(interface)
    camera.start()

    print("[MAIN] Waiting for Go2 camera frames...")

    while True:
        try:
            frame = camera.get_latest_frame(timeout=1.0)

            if frame is None:
                print("[CAMERA] No frame received")
                continue

            if SHOW_CAMERA_WINDOW:
                preview = cv2.resize(frame, (320, 180))
                cv2.imshow("Go2 Camera Window", preview)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Stopped by q key.")
                    break

            image = go2_frame_to_pil(frame)

            result = run_phi_vision(
                image=image,
                model=model,
                processor=processor,
                device=device
            )

            print("\n=== Phi-3.5-Vision ROBOT OUTPUT ===")
            print(result)

            if SAVE_PNG:
                print(f"[PNG] Saved latest frame as: {PNG_FILENAME}")

            time.sleep(1)

        except KeyboardInterrupt:
            print("Stopped stream.")
            break

        except Exception as e:
            print("[ERROR]", e)
            time.sleep(0.5)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

# run = python3 go2_phi_vision.py eth0
