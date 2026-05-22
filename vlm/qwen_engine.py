from transformers import Qwen2_5_VLForConditionalGeneration
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from backend.camera_streamer import Go2CameraStreamer

import time
import cv2
from PIL import Image

import torch

print("Loading Qwen model...")

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto"
)

processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct"
)

print("Qwen loaded successfully.")


def run_qwen_with_frame(pil_image, prompt):
    """
    Accepts a PIL Image directly from the camera stream,
    bypassing the need for a file path.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": pil_image, # Pass the PIL Image object directly
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
            max_new_tokens=128,
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

def main():
    # 1. Initialize the streamer with your target network interface
    # (e.g., 'enp2s0' for wired, 'wlan0' for Wi-Fi)
    network_interface = "enp2s0" 
    streamer = Go2CameraStreamer(interface=network_interface)
    streamer.start()

    print("\nStarting Qwen2.5-VL Inference Loop. Press Ctrl+C to stop.")
    prompt = "Describe what the robot sees in front of it in one short sentence."

    try:
        while True:
            # 2. Grab the freshest frame from the background queue (blocks up to 5s)
            frame = streamer.get_latest_frame(timeout=5)
            
            if frame is not None:
                # 3. OpenCV reads in BGR format; Qwen expects RGB format
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 4. Convert the NumPy array into a PIL Image object
                pil_image = Image.fromarray(rgb_frame)

                # 5. Run inference and track execution time
                start_time = time.time()
                description = run_qwen_with_frame(pil_image, prompt)
                latency = time.time() - start_time
                
                print(f"\n--- Qwen Output ({latency:.2f}s) ---")
                print(description)
                print("----------------------------------\n")
                
                # Optional: Show what the camera is tracking in real-time
                cv2.imshow("Go2 Front Camera Feed", frame)
                if cv2.waitKey(1) == 27: # Press 'ESC' key to break out
                    break
            else:
                print("Waiting for camera frames from DDS...")

    except KeyboardInterrupt:
        print("\nStopping VLM inference loop...")
    finally:
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()