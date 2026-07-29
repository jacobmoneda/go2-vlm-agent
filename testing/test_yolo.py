# test_yolo.py
from PIL import Image
from backend.objectDetection.yolo_engine import get_detections, run_yolo_with_frame

# test with a static image first
img = Image.open("C:/Users/jmone/OneDrive/Documents/GitHub/go2-vlm-agent/test_image.jpg")

print("=== Detections ===")
detections = get_detections(img)
for d in detections:
    print(f"{d['label']}: confidence={d['confidence']:.2f} center=({int(d['box_center_x'])}, {int(d['box_center_y'])}) height={int(d['box_height'])}px")

print("\n=== Follow Command ===")
command = run_yolo_with_frame(img, "follow the person")
print(command)