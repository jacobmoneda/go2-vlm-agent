from ultralytics import YOLO
from PIL import Image
import numpy as np

MODEL_PATH = "/home/unitree/go2-vlm-agent/models/yolo11n.pt"

FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
CENTER_X = FRAME_WIDTH // 2   # 960
CENTER_Y = FRAME_HEIGHT // 2  # 540
DEAD_ZONE = 60                # pixels either side of center before turning
CLOSE_THRESHOLD = 600          # bounding box height in pixels — stop if target this close


print("[YOLO] Loading model...")
model = YOLO(MODEL_PATH)
print("[YOLO] Model loaded successfully.")


def get_detections(pil_image: Image.Image) -> list:
    """
    Run YOLO on a PIL image and return all detections as a list of dicts.
    Each dict contains: label, confidence, box_center_x, box_center_y, box_height, xyxy
    """
    frame = np.array(pil_image)
    results = model(frame, verbose=False)

    detections = []
    for box in results[0].boxes:
        label = model.names[int(box.cls)]
        confidence = float(box.conf)
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        box_center_x = (x1 + x2) / 2
        box_center_y = (y1 + y2) / 2
        box_height = y2 - y1

        detections.append({
            "label": label,
            "confidence": confidence,
            "box_center_x": box_center_x,
            "box_center_y": box_center_y,
            "box_height": box_height,
            "xyxy": [x1, y1, x2, y2]
        })

    return detections


def get_follow_command(pil_image: Image.Image, target_class: str = "person") -> dict:
    """
    Analyse the image for a target object and return a movement command dict.
    Returns action, target_visible, target_description, confidence, and reasoning.
    """
    detections = get_detections(pil_image)

    # filter for target class above confidence threshold
    targets = [d for d in detections if d["label"] == target_class and d["confidence"] > 0.5]

    if not targets:
        return {
            "action": "search",
            "target_visible": False,
            "target_description": None,
            "confidence": 0.0,
            "reasoning": f"No {target_class} detected, searching"
        }

    # pick the highest confidence target
    target = max(targets, key=lambda d: d["confidence"])

    offset_x = target["box_center_x"] - CENTER_X
    box_height = target["box_height"]

    # determine action based on position and distance
    if box_height >= CLOSE_THRESHOLD:
        action = "stop"
        reasoning = f"{target_class} is close, stopping"
    elif abs(offset_x) > DEAD_ZONE:
        if offset_x > 0:
            action = "move_right"
            reasoning = f"{target_class} is to the right, turning right"
        else:
            action = "move_left"
            reasoning = f"{target_class} is to the left, turning left"
    else:
        action = "move_forward"
        reasoning = f"{target_class} is centered and far, moving forward"

    return {
        "action": action,
        "target_visible": True,
        "target_description": f"{target_class} at ({int(target['box_center_x'])}, {int(target['box_center_y'])})",
        "confidence": round(target["confidence"], 2),
        "reasoning": reasoning
    }


def run_yolo_with_frame(pil_image: Image.Image, user_command: str) -> dict:
    """
    Main entry point. Parses user command to extract target class,
    then runs follow logic. Returns a command dict compatible with Decision Logic.
    """
    # extract target class from user command
    target_class = extract_target_class(user_command)
    return get_follow_command(pil_image, target_class)


def extract_target_class(user_command: str) -> str:
    """
    Extract the target object class from the user command.
    Defaults to 'person' if no known class is found.
    """
    known_classes = [
        "person", "chair", "cup", "laptop", "backpack",
        "bottle", "dog", "cat", "phone", "book", "bag"
    ]
    command_lower = user_command.lower()
    for cls in known_classes:
        if cls in command_lower:
            return cls
    return "person"  # default target