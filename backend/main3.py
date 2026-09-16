# backend/main3.py
# pre-processing, routing, and 3-path execution (direct, phi, yolo-follow)
import threading
import io
import time
import uvicorn
from PIL import Image

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
ChannelFactoryInitialize(0, "eth0")

from backend.camera.go2_camera import Go2Camera
from backend.vlm.phi_engine import run_phi_with_frame
from backend.objectDetection.yolo_engine import get_detections
from backend.robotControl.robot_control import execute_action
from backend.decision_logic import follow_target
from backend.utils.input_processor import preprocess_prompt, InvalidPromptError
from backend.utils.command_router import parse_command
from backend.shared_state import shared_state
from backend.server import app

def warmup_ollama():
    import requests
    import time

    print("[Main] Warming up Ollama...")
    start = time.time()

    try:
        r = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "phi3-fast",
                "prompt": "",
                "stream": False,
                "keep_alive": -1,
            },
            timeout=600,
        )

        r.raise_for_status()
        d = r.json()

        print(f"[Main] Wall time: {time.time() - start:.2f}s")
        print(f"[Main] load_duration:        {d.get('load_duration', 0)/1e9:.2f}s")
        print(f"[Main] prompt_eval_duration: {d.get('prompt_eval_duration', 0)/1e9:.2f}s")
        print(f"[Main] eval_duration:        {d.get('eval_duration', 0)/1e9:.2f}s")
        print(f"[Main] total_duration:       {d.get('total_duration', 0)/1e9:.2f}s")

    except Exception as e:
        print(f"[Main] warmup failed: {e}")

def perception_loop(camera: Go2Camera):
    print("[Main] Perception loop started.")

    last_processed_prompt = None

    while True:
        if not camera.is_ready():
            time.sleep(0.05)
            continue

        raw_prompt = shared_state.user_prompt

        # ------------------------------------------------
        # PRE-PROCESSING
        # ------------------------------------------------

        # skip if no prompt or same as last processed
        if not raw_prompt or raw_prompt == last_processed_prompt:
            time.sleep(0.05)
            continue

        # start timer when new prompt is detected
        t_start = time.time()

        # Step 1 — input_processor: guard against injection and invalid commands
        try:
            prompt = preprocess_prompt(raw_prompt)
        except InvalidPromptError as e:
            print(f"[Main] Invalid prompt: {e}")
            last_processed_prompt = raw_prompt
            continue

        # preprocessing
        t_preprocess = time.time()

        # Step 2 — command_router: LLM determines intent
        print(f"[Main] Routing prompt: '{prompt}'")
        parsed = parse_command(prompt)

        needs_vision = parsed.get("needs_vision", False)
        is_follow = parsed.get("is_follow_command", False)
        action = parsed.get("action", "stop")
        confidence = parsed.get("confidence", 0.0)

        print(f"[Router] needs_vision={needs_vision} | is_follow={is_follow} | action={action} | confidence={confidence:.2f}")

        # routing
        t_route = time.time()

        # ------------------------------------------------
        # COMMAND EXECUTION
        # ------------------------------------------------

        # --- Path 1: No vision needed — execute instantly ---
        if not needs_vision and not is_follow:
            if confidence >= 0.6 and action:
                print(f"[Main] Direct execution: {action}")
                execute_action(action)
                shared_state.latest_result = {
                    "engine": "direct",
                    "action": action,
                    "reasoning": parsed.get("reasoning", "")
                }
            else:
                print(f"[Main] Low confidence ({confidence:.2f}) — stopping")
                execute_action("stop")

            last_processed_prompt = raw_prompt
            t_execute = time.time()
            print(f"[Latency] preprocess={t_preprocess-t_start:.3f}s | route={t_route-t_preprocess:.3f}s | execute={t_execute-t_route:.3f}s | total={t_execute-t_start:.3f}s")

        # --- Path 2: Vision needed — run Phi (one shot) ---
        elif needs_vision and not is_follow:
            print("[Main] Running Phi-3.5 for vision-dependent command...")
            t0 = time.time()

            frame_bytes = camera.get_frame_bytes()
            pil_image = Image.open(io.BytesIO(frame_bytes)).convert("RGB")

            result = run_phi_with_frame(pil_image, prompt)
            elapsed = time.time() - t0

            print(f"[Phi] {elapsed:.2f}s | {result}")
            shared_state.latest_result = {"engine": "phi", "result": result}

            # parse phi output and execute
            if isinstance(result, dict):
                phi_action = result.get("action", "stop")
                phi_confidence = result.get("confidence", 0.0)
                if phi_confidence >= 0.6 and phi_action:
                    execute_action(phi_action)
                else:
                    execute_action("stop")
            else:
                execute_action("stop")

            last_processed_prompt = raw_prompt
            t_execute = time.time()
            print(f"[Latency] preprocess={t_preprocess-t_start:.3f}s | route={t_route-t_preprocess:.3f}s | execute={t_execute-t_route:.3f}s | total={t_execute-t_start:.3f}s")
            
        # --- Path 3: Follow command — run YOLO continuously ---
        elif is_follow:
            print("[Main] Starting YOLO follow loop...")

            # extract target from prompt — default to person
            target_class = parsed.get("target", "person") or "person"

            while True:
                # check if user has changed the prompt
                current_prompt = shared_state.user_prompt
                if current_prompt != raw_prompt:
                    print("[Main] Prompt changed — exiting follow loop")
                    execute_action("stop")
                    break

                if not camera.is_ready():
                    time.sleep(0.05)
                    continue

                t0 = time.time()

                frame_bytes = camera.get_frame_bytes()
                pil_image = Image.open(io.BytesIO(frame_bytes)).convert("RGB")

                # run YOLO follow logic
                follow_target(target_class, pil_image)

                elapsed = time.time() - t0
                shared_state.latest_result = {
                    "engine": "yolo",
                    "target": target_class,
                    "elapsed": round(elapsed, 2)
                }

                time.sleep(0.1)  # ~10fps follow loop

            last_processed_prompt = shared_state.user_prompt
            t_execute = time.time()
            print(f"[Latency] preprocess={t_preprocess-t_start:.3f}s | route={t_route-t_preprocess:.3f}s | execute={t_execute-t_route:.3f}s | total={t_execute-t_start:.3f}s")
        


def main():
    camera = Go2Camera(network_interface="eth0")
    camera.start()
    shared_state.camera = camera

    warmup_ollama()

    perception_thread = threading.Thread(
        target=perception_loop,
        args=(camera,),
        daemon=True,
        name="PerceptionThread"
    )
    perception_thread.start()

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()