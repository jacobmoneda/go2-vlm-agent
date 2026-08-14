# testing/test_adaptive_follow.py
import io
import time
from PIL import Image
from unitree_sdk2py.core.channel import ChannelFactoryInitialize

ChannelFactoryInitialize(0, "eth0")

from backend.camera.go2_camera import Go2Camera
from backend.objectDetection.yolo_engine import get_detections
from backend.robotControl.robot_control import execute_action
from backend.adaptivePolicy.adaptive_policy import AdaptiveFollowPolicy

TARGET_CLASS = "person"
SAVE_INTERVAL = 50   # save policy every 50 frames

print("[Adaptive] Starting adaptive follow test...")
print("[Adaptive] Press Ctrl+C to stop\n")

policy = AdaptiveFollowPolicy()
camera = Go2Camera(network_interface="eth0")
camera.start()

frame_count = 0

try:
    while True:
        if not camera.is_ready():
            time.sleep(0.1)
            continue

        frame_bytes = camera.get_frame_bytes()
        pil_image = Image.open(io.BytesIO(frame_bytes)).convert("RGB")

        detections = get_detections(pil_image)
        targets = [d for d in detections if d["label"] == TARGET_CLASS and d["confidence"] > 0.5]

        if not targets:
            target_visible = False
            offset_x = 0
            box_height = 0
        else:
            target_visible = True
            target = max(targets, key=lambda d: d["box_height"])
            offset_x = target["box_center_x"] - 960
            box_height = target["box_height"]

        # get action from adaptive policy
        action = policy.get_action(offset_x, box_height, target_visible)

        # update policy based on outcome
        policy.update(action, offset_x, box_height, target_visible)

        # execute on robot
        execute_action(action)

        print(f"[Adaptive] action={action} | offset={int(offset_x)} | height={int(box_height)} | dead_zone={policy.dead_zone} | close_threshold={policy.close_threshold}")

        frame_count += 1

        # save periodically
        if frame_count % SAVE_INTERVAL == 0:
            policy.save()
            stats = policy.get_stats()
            print(f"[Stats] centring={stats['centring_accuracy_%']}% | retention={stats['target_retention_%']}% | adaptations={stats['adaptations_made']}")

        time.sleep(0.1)

except KeyboardInterrupt:
    print("\n[Adaptive] Saving policy and printing stats...")
    policy.save()
    execute_action("stop")

    stats = policy.get_stats()
    print("\n=== Session Stats ===")
    print(f"Total frames:        {stats['total_frames']}")
    print(f"Centring accuracy:   {stats['centring_accuracy_%']}%")
    print(f"Target retention:    {stats['target_retention_%']}%")
    print(f"Final dead_zone:     {stats['current_dead_zone']}")
    print(f"Final close_thresh:  {stats['current_close_threshold']}")
    print(f"Adaptations made:    {stats['adaptations_made']}")
    if stats['adaptation_log']:
        print("\nAdaptation log:")
        for entry in stats['adaptation_log']:
            print(f"  - {entry}")