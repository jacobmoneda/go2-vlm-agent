# backend/decision_logic.py

import io
import time
from PIL import Image
import time
from backend.objectDetection.yolo_engine import get_detections
from backend.robotControl.robot_control import execute_action
from backend.shared_state import shared_state 

FRAME_WIDTH = 1920
CENTER_X = FRAME_WIDTH // 2  # 960
DEAD_ZONE = 60
CONFIDENCE_THRESHOLD = 0.5

last_status = None

def send_status(message):

    global last_status

    if message != last_status:
        print(message)
        shared_state.status_queue.put(message)
        last_status = message

# calibrated distance thresholds — based on bounding box height at 1m
TARGET_HEIGHT_1M = 800
DISTANCE_TOLERANCE = 60
FAR_THRESHOLD = TARGET_HEIGHT_1M - DISTANCE_TOLERANCE    # 740 — move forward
CLOSE_THRESHOLD = TARGET_HEIGHT_1M + DISTANCE_TOLERANCE  # 860 — move backward


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
        time.sleep(0.1)
        return

    # pick nearest target (largest bounding box = closest)
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
    # priority 2 — target too far, move forward
    elif box_height < FAR_THRESHOLD:
        print("[Decision] Target too far — moving FORWARD")
        execute_action("move_forward")
    # priority 3 — target too close, move backward
    elif box_height > CLOSE_THRESHOLD:
        print("[Decision] Target too close — moving BACKWARD")
        execute_action("move_backward")
    # priority 4 — target approximately 1m away
    else:
        print("[Decision] Target ~1m away — stopping")
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