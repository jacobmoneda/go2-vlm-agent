# testing/test_palm_detection.py
import io
import time
from PIL import Image
from unitree_sdk2py.core.channel import ChannelFactoryInitialize

ChannelFactoryInitialize(0, "eth0")

from backend.camera.go2_camera import Go2Camera
from ultralytics import YOLO
import numpy as np

PALM_MODEL_PATH = "/home/unitree/models/palm_detector.pt"
CONFIDENCE_THRESHOLD = 0.6

print("[Palm] Loading palm detection model...")
model = YOLO(PALM_MODEL_PATH)
print("[Palm] Model loaded. Starting detection...\n")

camera = Go2Camera(network_interface="eth0")
camera.start()

try:
    while True:
        if not camera.is_ready():
            time.sleep(0.1)
            continue

        frame_bytes = camera.get_frame_bytes()
        pil_image = Image.open(io.BytesIO(frame_bytes)).convert("RGB")
        frame = np.array(pil_image)

        results = model(frame, verbose=False)

        detections = []
        for box in results[0].boxes:
            label = model.names[int(box.cls)]
            confidence = float(box.conf)
            if confidence >= CONFIDENCE_THRESHOLD:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append({
                    "label": label,
                    "confidence": confidence,
                    "center": (int((x1 + x2) / 2), int((y1 + y2) / 2))
                })

        if detections:
            for d in detections:
                print(f"[Palm] DETECTED — label={d['label']} | confidence={d['confidence']:.2f} | center={d['center']}")
        else:
            print("[Palm] No palm detected")

        time.sleep(0.2)

except KeyboardInterrupt:
    print("\n[Palm] Test stopped.")