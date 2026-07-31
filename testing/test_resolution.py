# testing/test_resolution.py
import io
import time
from PIL import Image
from backend.camera.go2_camera import Go2Camera

camera = Go2Camera(network_interface="eth0")
camera.start()

while not camera.is_ready():
    time.sleep(0.1)

frame_bytes = camera.get_frame_bytes()
pil_image = Image.open(io.BytesIO(frame_bytes)).convert("RGB")
print(f"Resolution: {pil_image.size}")