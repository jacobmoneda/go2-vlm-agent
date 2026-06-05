# backend/main.py
import threading
import uvicorn
from camera.go2_camera import Go2Camera
from vlm.qwen_engine import run_qwen_with_frame
from shared_state import shared_state
from server import app


def perception_loop(camera: Go2Camera):
    print("[Main] Perception loop started.")
    while True:
        if not camera.is_ready():
            continue

        frame_bytes = camera.get_frame_bytes()

        result = run_qwen_with_frame(frame_bytes, shared_state.latest_prompt)
        shared_state.latest_result = result
        print("[Perception]", result)

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