# test_go2_camera.py (run from project root)
import time
from backend.camera.go2_camera import Go2Camera

cam = Go2Camera()  # pass "eth0" or interface name if needed
cam.start()

print("Waiting for first frame...")
while not cam.is_ready():
    time.sleep(0.1)

print("First frame received.")

# Grab 5 frames, save each
for i in range(5):
    raw = cam.get_frame_bytes()
    if raw:
        with open(f"./test_frame_{i}.jpg", "wb") as f:
            f.write(raw)
        print(f"Saved test_frame_{i}.jpg")
    time.sleep(0.5)

cam.stop()