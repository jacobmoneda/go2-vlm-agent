import sys
import time
import queue
import cv2
import torch
import numpy as np
from PIL import Image

from transformers import AutoProcessor, AutoModelForImageTextToText

from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_go.msg.dds_ import Go2FrontVideoData_


# =========================
# Settings
# =========================

DEFAULT_INTERFACE = "eth0"

MODEL_ID = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"

# If you run from PuTTY / SSH and cannot open a window, set this to False
SHOW_CAMERA_WINDOW = True

# For hand/finger recognition, 224x224 is better than 50x50
IMAGE_SIZE = (224, 224)

# Save latest Go2 camera image as PNG
SAVE_PNG = True
PNG_FILENAME = "latest_go2_frame.png"


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
                # If queue is full, drop the oldest frame to keep latency low
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
# Load SmolVLM2 Model
# =========================

device = "cuda" if torch.cuda.is_available() else "cpu"
print("[AI] Device:", device)

processor = AutoProcessor.from_pretrained(MODEL_ID)

model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto" if device == "cuda" else None
)

if device == "cpu":
    model = model.to("cpu")

print("[AI] SmolVLM2 loaded successfully")


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
Only describe the visible hand gesture.

Focus on these gestures:
- five fingers spread up
- index finger pointing up
- finger pointing forward
- peace sign
- three fingers raised
- fist raised
- thumbs up
- no clear hand gesture

Use this exact format:
Type:
Hand gesture:
Location:
"""


# =========================
# Convert Go2 Frame to PIL Image
# =========================

def go2_frame_to_pil(frame_bgr):
    """
    Go2 camera frame from OpenCV is BGR.
    SmolVLM2 needs PIL RGB image.
    """

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb).convert("RGB")

    # Resize for model input
    image = image.resize(IMAGE_SIZE)

    # Save as PNG for checking/debugging
    if SAVE_PNG:
        image.save(PNG_FILENAME, format="PNG")

    return image


# =========================
# SmolVLM2 Vision Function
# =========================

def run_smolvlm2(image):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": ROBOT_PROMPT}
            ]
        }
    ]

    # Main method for new Transformers version
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

    # Backup method for older Transformers version
    except TypeError:
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

    # Keep only newly generated AI answer
    generated_ids_trimmed = generated_ids[:, inputs["input_ids"].shape[-1]:]

    answer = processor.decode(
        generated_ids_trimmed[0],
        skip_special_tokens=True
    )

    return answer.strip()


# =========================
# Main Loop
# =========================

def main():
    if len(sys.argv) >= 2:
        interface = sys.argv[1]
    else:
        interface = DEFAULT_INTERFACE

    print(f"[MAIN] Using network interface: {interface}")

    camera = Go2CameraStreamer(interface)
    camera.start()

    print("[MAIN] Waiting for Go2 camera frames...")

    while True:
        try:
            frame = camera.get_latest_frame(timeout=1.0)

            if frame is None:
                print("[CAMERA] No frame received")
                continue

            # Show Go2 camera preview on computer screen
            if SHOW_CAMERA_WINDOW:
                preview = cv2.resize(frame, (320, 180))
                cv2.imshow("Go2 Camera Window", preview)

                # Press q to stop
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Stopped by q key.")
                    break

            # Convert Go2 frame to PIL image
            image = go2_frame_to_pil(frame)

            # Run SmolVLM2 vision model
            result = run_smolvlm2(image)

            print("\n=== SmolVLM2 ROBOT VISION OUTPUT ===")
            print(result)

            if SAVE_PNG:
                print(f"[PNG] Saved latest Go2 frame as: {PNG_FILENAME}")

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
        
