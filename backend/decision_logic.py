# backend/decision_logic.py
from PIL import Image
import time
from backend.objectDetection.yolo_engine import get_detections
from backend.robotControl.robot_control import execute_action

FRAME_WIDTH = 1920
CENTER_X = FRAME_WIDTH // 2
DEAD_ZONE = 60
CONFIDENCE_THRESHOLD = 0.2

# calibrated distance thresholds — based on bounding box height at 1m
TARGET_HEIGHT_1M = 800
DISTANCE_TOLERANCE = 60
FAR_THRESHOLD = TARGET_HEIGHT_1M - DISTANCE_TOLERANCE    # 740 — move forward
CLOSE_THRESHOLD = TARGET_HEIGHT_1M + DISTANCE_TOLERANCE  # 860 — move backward


def follow_target(target_class: str, pil_image: Image.Image):
    detections = get_detections(pil_image)
    print(f"[YOLO] Detections: {[(d['label'], round(d['confidence'], 2)) for d in detections]}")

    targets = [d for d in detections if d["label"] == target_class and d["confidence"] > CONFIDENCE_THRESHOLD]

    if not targets:
        print(f"[Decision] No {target_class} detected — stopping")
        execute_action("stop")
        time.sleep(0.1)
        return

    # pick nearest target (largest bounding box = closest)
    target = max(targets, key=lambda d: d["box_height"])
    offset_x = target["box_center_x"] - CENTER_X
    box_height = target["box_height"]

    print(f"[Decision] offset={int(offset_x)} | height={int(box_height)}px | confidence={target['confidence']:.2f}")

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


def detect_object(target_class: str, pil_image: Image.Image) -> bool:
    detections = get_detections(pil_image)
    targets = [d for d in detections if d["label"] == target_class and d["confidence"] > CONFIDENCE_THRESHOLD]
    if not targets:
        print(f"[Detection] No {target_class} detected.")
        return False
    target = max(targets, key=lambda d: d["confidence"])
    print(f"[Detection] {target_class} detected | confidence={target['confidence']:.2f}")
    return True