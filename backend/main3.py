#main3.py
# integrating 2 routes: vision dependent command and instant llm-based command
import threading
import time
import uvicorn
from backend.camera.go2_camera import Go2Camera

from backend.decision_logic import run_task
from backend.utils.input_processor import process_input, InvalidPromptError

from backend.vlm.phi_engine import run_phi_with_frame
from backend.objectDetection.yolo_engine import run_yolo_with_frame

from backend.shared_state import shared_state
from backend.server import app
from backend.nlp.command_router import parse_command

def perception_loop(camera):
    while True:
        if not camera.is_ready():
            continue
        
        frame_bytes = camera.get_frame_bytes()
        pil_image = Image.open(io.BytesIO(frame_bytes)).convert("RGB")
        prompt = shared_state.latest_prompt

        # Step 1 — LLM parses command instantly
        parsed = parse_command(prompt)

        # Step 2 — route based on parsed intent
        if parsed.get("is_follow_command"):
            # YOLO handles real-time tracking
            result = run_yolo_with_frame(pil_image, prompt)
            decide_and_act(result, execute_action)

        elif parsed.get("needs_vision"):
            # Phi handles vision-dependent commands (20s)
            result = run_phi_with_frame(pil_image, prompt)
            decide_and_act(result, execute_action)

        else:
            # LLM already determined action — execute immediately
            action = parsed.get("action", "stop")
            confidence = parsed.get("confidence", 0.0)
            if confidence >= 0.6 and action:
                print(f"[LLM] Direct action: {action} | {parsed.get('reasoning')}")
                execute_action(action)

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