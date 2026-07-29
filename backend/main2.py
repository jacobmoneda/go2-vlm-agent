# backend/main2.py
import threading
import io
import time
import uvicorn
from PIL import Image
from backend.camera.go2_camera import Go2Camera
from backend.objectDetection.yolo_engine import run_yolo_with_frame, get_detections, get_follow_command
from backend.shared_state import shared_state
from backend.server import app


def perception_loop(camera: Go2Camera):
    print("[Main] Perception loop started.")
    while True:
        if not camera.is_ready():
            continue

        t0 = time.time()

        frame_bytes = camera.get_frame_bytes()
        pil_image = Image.open(io.BytesIO(frame_bytes)).convert("RGB")

        #result = run_qwen_with_frame(pil_image, shared_state.latest_prompt)
        result = run_yolo_with_frame(pil_image, shared_state.latest_prompt)
        result = get_follow_command(pil_image, )
        #result = run_smolvlm_with_frame(pil_image, shared_state.latest_prompt)
        shared_state.latest_result = result

        elapsed = time.time() - t0
        print("[Prompt] ", shared_state.latest_prompt)
        print(f"[Perception] {elapsed:.2f}s | {result}")

def main():
    # Start camera
    camera = Go2Camera(network_interface="eth0")
    camera.start()

    # Start perception loop in background thread
    perception_thread = threading.Thread(
        target=perception_loop,
        args=(camera,),
        daemon=True,
        name="PerceptionThread"
    )
    perception_thread.start()

    # Start WebSocket server (blocks)
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
