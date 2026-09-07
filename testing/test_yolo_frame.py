# testing/test_yolo_static.py
from ultralytics import YOLO
from PIL import Image
import numpy as np

model = YOLO("/home/unitree/models/yolo11n.pt")
model.to("cpu")

img = Image.open("/home/unitree/go2-vlm-agent/debug_frame.jpg").convert("RGB")
frame = np.array(img)

results = model(frame, verbose=True, device="cpu")

print(f"\nAll detections:")
for box in results[0].boxes:
    label = model.names[int(box.cls)]
    confidence = float(box.conf)
    print(f"  {label}: {confidence:.2f}")