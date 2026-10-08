# backend/decision_logic.py

import io
import time
from PIL import Image
import time
from backend.objectDetection.yolo_engine import get_detections, get_tracked_detections
from backend.robotControl.robot_control import execute_action
from backend.shared_state import shared_state 

FRAME_WIDTH = 1920
CENTER_X = FRAME_WIDTH // 2  # 960
DEAD_ZONE = 60
CONFIDENCE_THRESHOLD = 0.5

last_status = None
MAX_LOST_FRAMES = 100
locked_target_id = None
locked_target_class = None
lost_target_frames = 0


def reset_target_lock():
    """Start a fresh selection when entering a new follow/navigation command."""
    global locked_target_id, locked_target_class, lost_target_frames
    if locked_target_id is not None:
        print(f"[Target] Clearing target lock ID={locked_target_id}")
    locked_target_id = None
    locked_target_class = None
    lost_target_frames = 0


def lock_target(target_class, track_id):
    global locked_target_id, locked_target_class, lost_target_frames
    locked_target_id = track_id
    locked_target_class = target_class
    lost_target_frames = 0
    print(f"[Target] Locked class={target_class} ID={track_id}")

def send_status(message):

    global last_status

    if message != last_status:
        print(message)
        shared_state.status_queue.put(message)
        last_status = message

# calibrated distance thresholds — based on bounding box height at 1m
TARGET_HEIGHT_1M = 600
DISTANCE_TOLERANCE = 60
FAR_THRESHOLD = TARGET_HEIGHT_1M - DISTANCE_TOLERANCE    # 740 — move forward
CLOSE_THRESHOLD = TARGET_HEIGHT_1M + DISTANCE_TOLERANCE  # 860 — move backward


def follow_target(target_class: str, pil_image: Image.Image):
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
    global lost_target_frames

    detections = get_tracked_detections(pil_image)
    print(
        f"[Tracks] {target_class}: "
        f"{[(d['track_id'], round(d['confidence'], 2)) for d in detections if d['label'] == target_class]}"
    )

    # A timeout is latched until the next command explicitly resets the lock.
    if locked_target_id is not None and lost_target_frames >= MAX_LOST_FRAMES:
        send_status(f"[Target] Selected {locked_target_class} ID={locked_target_id} lost — issue a new command")
        execute_action("stop")
        return

    # filter for target class above confidence threshold
    targets = [
        d for d in detections
        if d["label"] == target_class
        and d["confidence"] >= CONFIDENCE_THRESHOLD
        and d["track_id"] is not None
    ]

    if locked_target_id is None:
        if not targets:
            send_status(f"[Decision] No tracked {target_class} visible — stopping")
            execute_action("stop")
            time.sleep(0.1)
            return
        # The largest-box heuristic is used only for initial selection.
        target = max(targets, key=lambda d: d["box_height"])
        print(f"[Target] Initial selection: {target_class} ID={target['track_id']}")
        lock_target(target_class, target["track_id"])
    else:
        target = next(
            (d for d in targets
             if d["track_id"] == locked_target_id
             and d["label"] == locked_target_class),
            None,
        )
        if target is None:
            lost_target_frames += 1
            execute_action("stop")
            print(f"[Target] Lost {locked_target_class} ID={locked_target_id} ({lost_target_frames}/{MAX_LOST_FRAMES})")
            if lost_target_frames >= MAX_LOST_FRAMES:
                send_status(f"[Target] Selected {locked_target_class} ID={locked_target_id} lost — issue a new command")
            return
        lost_target_frames = 0

    offset_x = target["box_center_x"] - CENTER_X
    box_height = target["box_height"]

    print(
        f"[Target] Following {target_class} ID={locked_target_id} "
        f"| x={int(target['box_center_x'])} "
        f"| offset={int(offset_x)} "
        f"| height={int(box_height)}px "
        f"| confidence={target['confidence']:.2f}"
    )

    # priority 1 — turn to centre target first
    if offset_x > DEAD_ZONE:
        send_status("[Follow Decision] Turning RIGHT")
        execute_action("turn_right")

    elif offset_x < -DEAD_ZONE:
        send_status("[Follow Decision] Turning LEFT")
        execute_action("turn_left")
    # priority 2 — target too far, move forward
    elif box_height < FAR_THRESHOLD:
        send_status("[Follow Decision] Target too far — moving FORWARD")
        execute_action("move_forward")
    # priority 3 — target too close, move backward
    elif box_height > CLOSE_THRESHOLD:
        send_status("[Follow Decision] Target too close — moving BACKWARD")
        execute_action("move_backward")
    # priority 4 — target approximately 1m away
    else:
        send_status("[Follow Decision] Target ~1m away — stopping")
        execute_action("stop")


def detect_object(target_class: str, pil_image: Image.Image) -> bool:    
    """
    Check whether YOLO can currently detect a requested object.

    Does NOT move the robot.
    Returns True if the object is detected and False if it is not.
    """

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
    print(f"[Detection] {target_class} detected | confidence={target['confidence']:.2f}")

    send_status(
        f"[Detection] {target_class} detected "
        f"({target['confidence'] * 100:.0f}% confidence)"
)

    return True
