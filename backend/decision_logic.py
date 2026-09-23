# backend/decision_logic.py

import io
import time
from PIL import Image

from backend.camera.go2_camera import Go2Camera
from backend.objectDetection.yolo_engine import get_detections
from backend.robotControl.robot_control import execute_action
from backend.utils.input_processor import process_input, InvalidPromptError
from backend.shared_state import shared_state 

FRAME_WIDTH = 1920
CENTER_X = FRAME_WIDTH // 2  # 960
DEAD_ZONE = 60
CLOSE_THRESHOLD = 600
CONFIDENCE_THRESHOLD = 0.5

last_status = None

def send_status(message):

    global last_status

    if message != last_status:
        print(message)
        shared_state.status_queue.put(message)
        last_status = message


def follow_target(target_class, camera):
    """
    Follow / track a target using YOLO.

    This function:
    - gets the current camera frame
    - runs YOLO
    - finds the requested target
    - turns left/right to centre target
    - moves forward if centred and target is far
    - stops if centred and target is close
    """

    if not camera.is_ready():
        return

    # get frame from camera
    frame_bytes = camera.get_frame_bytes()
    pil_image = Image.open(io.BytesIO(frame_bytes)).convert("RGB")

    # run YOLO
    detections = get_detections(pil_image)

    # filter for target class above confidence threshold
    targets = [
        d for d in detections
        if d["label"] == target_class
        and d["confidence"] > CONFIDENCE_THRESHOLD
    ]

    # if target cannot be seen, stop
    if not targets:
        send_status(f"[Decision] No {target_class} detected — stopping")
        execute_action("stop")
        return

    # pick nearest target — largest bounding box = closest
    target = max(targets, key=lambda d: d["box_height"])

    offset_x = target["box_center_x"] - CENTER_X
    box_height = target["box_height"]

    print(
        f"[Decision] Target={target_class} "
        f"| x={int(target['box_center_x'])} "
        f"| offset={int(offset_x)} "
        f"| height={int(box_height)}px "
        f"| confidence={target['confidence']:.2f}"
    )

    # priority 1 — turn to centre target first
    if offset_x > DEAD_ZONE:
        print("[Decision] Turning RIGHT")
        execute_action("turn_right")

    elif offset_x < -DEAD_ZONE:
        print("[Decision] Turning LEFT")
        execute_action("turn_left")

    # priority 2 — move forward if centred and far enough
    elif box_height < CLOSE_THRESHOLD:
        print("[Decision] Moving FORWARD")
        execute_action("move_forward")

    # priority 3 — stop if centred and close enough
    else:
        print("[Decision] Close enough — stopping")
        execute_action("stop")


def detect_object(target_class, camera):
    """
    Check whether YOLO can currently detect a requested object.

    Does NOT move the robot.
    Returns True if the object is detected and False if it is not.
    """

    if not camera.is_ready():
        print("[Detection] Camera is not ready.")
        return False

    frame_bytes = camera.get_frame_bytes()

    if frame_bytes is None:
        print("[Detection] No camera frame available.")
        return False

    pil_image = Image.open(io.BytesIO(frame_bytes)).convert("RGB")

    detections = get_detections(pil_image)

    targets = [
        d for d in detections
        if d["label"] == target_class
        and d["confidence"] > CONFIDENCE_THRESHOLD
    ]

    if not targets:
        send_status(f"[Detection] No {target_class} detected.")
        return False

    target = max(targets, key=lambda d: d["confidence"])

    send_status(
        f"[Detection] {target_class} detected "
        f"({target['confidence'] * 100:.0f}% confidence)"
)

    return True


def run_task(task, camera):
    """
    Takes the processed user task and decides
    which robot behaviour should run.
    """

    action = task["action"]
    target = task["target"]

    print(f"[Decision] Action: {action}")
    print(f"[Decision] Target: {target}")

    # -------------------------
    # FOLLOW / TRACK / WATCH
    # -------------------------
    if action in ["follow", "track", "watch"]:

        if target is None:
            print("[Decision] No target supplied — defaulting to person")
            target = "person"

        follow_target(target, camera)

    # -------------------------
    # OBJECT DETECTION
    # -------------------------
    elif action in ["detect", "see"]:

        if target is None:
            print("[Decision] No object specified.")
            return

        detected = detect_object(target, camera)

        if detected:
            send_status(f"[Response] Yes, I can see a {target}.")
        else:
            send_status(f"[Response] No, I cannot see a {target}.")

    elif action == "stop":
        send_status("[Action] Stopping")
        execute_action("stop")

    elif action == "sit":
        send_status("[Action] Sitting")
        execute_action("sit")

    elif action == "stand":
        send_status("[Action] Standing")
        execute_action("stand_up")

    elif action == "wave":
        send_status("[Action] Waving")
        execute_action("hello")

    elif action == "emote":
        send_status("[Action] Waving")
        execute_action("hello")

    elif action == "move":
        send_status("[Action] Moving forward")
        execute_action("move_forward")


    elif action in ["find", "search"]:
        send_status("[Response] Find/search is not implemented yet.")
        execute_action("stop")

    elif action == "go to":
        send_status("[Response] Go-to behaviour is not implemented yet.")
        execute_action("stop")

    elif action in ["describe", "look"]:
        send_status("[Response] Visual description is not connected yet.")
        execute_action("stop")

    else:
        send_status(f"[Error] Unknown action: '{action}' — stopping")
        execute_action("stop")
