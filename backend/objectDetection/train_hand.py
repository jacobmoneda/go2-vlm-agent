# train_palm.py
from ultralytics import YOLO

# start from pretrained yolo11n for faster convergence
model = YOLO("yolo11n.pt")

model.train(
    data="C:/Users/jmone/hand_training/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    name="palm_detector",
    project="C:/Users/jmone/palm_training/runs"
)