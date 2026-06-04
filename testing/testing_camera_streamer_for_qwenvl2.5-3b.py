import sys
import threading
import queue
import cv2
import numpy as np
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_go.msg.dds_ import Go2FrontVideoData_

# =========================
# Add Import commands
# =========================

from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


# =========================
# Add Settings
# =========================
DEFAULT_INTERFACE = "eth0"
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
SHOW_CAMERA_WINDOW = True

# Qwen image size
IMAGE_SIZE = (224, 224)

# Save Go2 camera image as PNG
SAVE_PNG = True
PNG_FILENAME = "latest_go2_frame.png"

class Go2CameraStreamer:
    def __init__(self, interface: str):
        self.interface = interface
        self.frame_queue = queue.Queue(maxsize=2)  # Low maxsize keeps latency down
        self.is_running = False
        self.sub = None

    def _video_callback(self, msg: Go2FrontVideoData_):
        if msg.video720p and len(msg.video720p) > 0:
            np_arr = np.frombuffer(bytes(msg.video720p), dtype=np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if frame is not None:
                # If queue is full, drop the oldest frame to prevent VLM lag
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.frame_queue.put_nowait(frame)

    def start(self):
        """Starts the DDS subscription loop in the background."""
        ChannelFactoryInitialize(0, self.interface)
        self.sub = ChannelSubscriber("/frontvideostream", Go2FrontVideoData_)
        self.sub.Init(self._video_callback, 10)
        self.is_running = True
        print(f"[SDK] Subscribed to camera stream on {self.interface}")

    def get_latest_frame(self, timeout=None):
        """Call this from your VLM loop to pull the freshest image array."""
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None
        

# =========================
# Add Load Qwen2.5-VL Model
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("[AI] Device:", device)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto" if device == "cuda" else None
)

if device == "cpu":
    model = model.to("cpu")

processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    min_pixels=128 * 28 * 28,
    max_pixels=256 * 28 * 28
)

print("[AI] Qwen2.5-VL model loaded")


# =========================
# Add Convert Go2 Frame to PIL PNG Image
# =========================
def go2_frame_to_pil_png(frame_bgr):
    """
    Go2/OpenCV frame is BGR.
    Qwen/PIL image needs RGB.
    This function converts the frame and saves it as PNG.
    """

    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # Convert to PIL image
    image = Image.fromarray(frame_rgb).convert("RGB")

    # Resize image for Qwen
    image = image.resize(IMAGE_SIZE)

    # Save as PNG
    if SAVE_PNG:
        image.save(PNG_FILENAME, format="PNG")

    return image

# =========================
# Add Qwen Vision Function
# =========================
def describe_robot_view(image):
    messages = [
        {
            "role": "user",
            "content": 
            s[
                {"type": "image"},
                {
                    "type": "text",
                    "text": """
                    Only report:
                    1. Type: object, human, or animal
                    2. Hand gesture: describe only the hands movement, finger movement, finger direction, and hand position
                    3. Location: indoor or outdoor
                    
                    Do NOT describe clothing, glasses, face details, room details, ceiling, lights, or background objects.
                    Do NOT describe the person's intention or activity, such as taking a selfie.
                    Only describe the visible hands gesture.
                    
                    Only focus on:
                    - holding up hand with five fingers spread up
                    - pointing with index finger up
                    - pointing with finger forward
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
                }
            ]
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt"
    )

    inputs = inputs.to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False
        )

    # Remove system/user/assistant prompt text
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    return output[0].strip()

# =========================
# Add Main Loop
# =========================
def main():
    # If no interface is typed, use eth0 automatically
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

            # Small camera preview window
            if SHOW_CAMERA_WINDOW:
                preview = cv2.resize(frame, IMAGE_SIZE)
                cv2.imshow("Go2 Camera Window", preview)

                # Press q to stop
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Stopped by q key.")
                    break

            # Convert Go2 frame to PIL image and save as PNG
            image = go2_frame_to_pil_png(frame)

            # Run Qwen vision
            result = describe_robot_view(image)

            print("\n=== ROBOT VISION OUTPUT ===")
            print(result)

            if SAVE_PNG:
                print(f"[PNG] Saved latest camera image as: {PNG_FILENAME}")

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

        # Run command = python3 go2_qwen_vision.py
