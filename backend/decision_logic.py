# backend/decision_logic.py
from PIL import Image

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
ChannelFactoryInitialize(0, "eth0")

from backend.objectDetection.yolo_engine import get_detections
from backend.robotControl.robot_control import execute_action

FRAME_WIDTH = 1920
CENTER_X = FRAME_WIDTH // 2
DEAD_ZONE = 60
CLOSE_THRESHOLD = 600
CONFIDENCE_THRESHOLD = 0.2


def follow_target(target_class: str, pil_image: Image.Image):
    detections = get_detections(pil_image)
    print(f"[YOLO] Detections: {[(d['label'], round(d['confidence'], 2)) for d in detections]}")

    targets = [d for d in detections if d["label"] == target_class and d["confidence"] > CONFIDENCE_THRESHOLD]

    if not targets:
        print(f"[Decision] No {target_class} detected — stopping")
        execute_action("stop")
        return

    target = max(targets, key=lambda d: d["box_height"])
    offset_x = target["box_center_x"] - CENTER_X
    box_height = target["box_height"]

    print(f"[Decision] offset={int(offset_x)} | height={int(box_height)}px | confidence={target['confidence']:.2f}")

    if offset_x > DEAD_ZONE:
        print("[Decision] Turning RIGHT")
        execute_action("turn_right")
    elif offset_x < -DEAD_ZONE:
        print("[Decision] Turning LEFT")
        execute_action("turn_left")
    elif box_height < CLOSE_THRESHOLD:
        print("[Decision] Moving FORWARD")
        execute_action("move_forward")
    else:
        print("[Decision] Close enough — stopping")
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