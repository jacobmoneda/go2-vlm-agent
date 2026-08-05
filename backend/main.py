# backend/main.py
import threading
import time
import uvicorn
from backend.camera.go2_camera import Go2Camera

from backend.decision_logic import run_task
from backend.utils.input_processor import process_input, InvalidPromptError

# CHOOSE A MODEL TO USE FOR VISION-LANGUAGE PERCEPTION
# Uncomment one of the following lines to select the model
from backend.vlm.qwen_engine import run_qwen_with_frame
from backend.vlm.smolvlm_engine import run_smolvlm_with_frame
from backend.vlm.phi_engine import run_phi_with_frame


from backend.shared_state import shared_state
from backend.server import app

def perception_loop(camera: Go2Camera):
    print("[Main] Perception loop started.")

    while True:
        if not camera.is_ready():
            continue

        t0 = time.time()


        if not prompt:
            time.sleep(0.1)
            continue

        try:
            task = process_input(prompt)

        except InvalidPromptError as error:
            print(f"[Main] Invalid prompt: {error}")
            time.sleep(0.1)
            continue

        print("[Prompt]", prompt)
        print("[Task]", task)

        run_task(task, camera)

        elapsed = time.time() - t0
        print(f"[Decision] {elapsed:.2f}s")

        time.sleep(0.1)


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