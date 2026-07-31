# testing/test_yolo_follow.py
# testing/test_yolo_follow.py
import io
import time
from PIL import Image
from unitree_sdk2py.core.channel import ChannelFactoryInitialize

# must initialise DDS before importing robot_control
ChannelFactoryInitialize(0, "eth0")

from backend.camera.go2_camera import Go2Camera
from backend.objectDetection.yolo_engine import get_detections
from backend.robotControl.robot_control import execute_action


FRAME_WIDTH = 1920
CENTER_X = FRAME_WIDTH // 2  # 960
DEAD_ZONE = 60
CLOSE_THRESHOLD = 600
TARGET_CLASS = "person" # change to test with other objects

print("[Test] Starting YOLO follow test...")
print(f"[Test] Target class: {TARGET_CLASS}")
print(f"[Test] Dead zone: center ± {DEAD_ZONE}px")
print("[Test] Press Ctrl+C to stop\n")

camera = Go2Camera(network_interface="eth0")
camera.start()

try:
    while True:
        if not camera.is_ready():
            time.sleep(0.1)
            continue
 
        # grab frame
        frame_bytes = camera.get_frame_bytes()
        pil_image = Image.open(io.BytesIO(frame_bytes)).convert("RGB")
 
        # run YOLO
        detections = get_detections(pil_image)
 
        # filter for target class
        targets = [d for d in detections if d["label"] == TARGET_CLASS and d["confidence"] > 0.5]
 
        if not targets:
            print("[YOLO] No target detected — stopping")
            execute_action("stop")
            time.sleep(0.1)
            continue
 
        # pick nearest target (largest bounding box = closest)
        target = max(targets, key=lambda d: d["box_height"])
 
        offset_x = target["box_center_x"] - CENTER_X
        box_height = target["box_height"]
 
        print(f"[YOLO] Target at x={int(target['box_center_x'])} | offset={int(offset_x)} | height={int(box_height)}px | confidence={target['confidence']:.2f}")
 
        # priority 1 — turn to centre target first
        if offset_x > DEAD_ZONE:
            print("[Action] Turning RIGHT")
            execute_action("turn_right")
        elif offset_x < -DEAD_ZONE:
            print("[Action] Turning LEFT")
            execute_action("turn_left")
        # priority 2 — move forward if centred and far enough
        elif box_height < CLOSE_THRESHOLD:
            print("[Action] Moving FORWARD")
            execute_action("move_forward")
        # priority 3 — stop if centred and close
        else:
            print("[Action] Close enough — stopping")
            execute_action("stop")
 
        time.sleep(0.1)  # small delay between commands
 
except KeyboardInterrupt:
    print("\n[Test] Stopped by user")
    execute_action("stop")