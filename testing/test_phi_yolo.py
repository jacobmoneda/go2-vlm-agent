import torch
from PIL import Image
from backend.objectDetection.yolo_engine import (
    get_model, get_detections, get_tracked_detections,
)
from backend.vlm.phi_engine import run_phi_with_frame, unload_phi

image = Image.open(
    "/home/unitree/go2-vlm-agent/images/live_yolo_frame.jpg"
).convert("RGB")

def check(stage):
    print("\n---", stage, "---")
    print("Default dtype:", torch.get_default_dtype())

    model = get_model()
    parameter = next(model.model.parameters())
    print("YOLO device/dtype:", parameter.device, parameter.dtype)

    detections = get_detections(image)
    print("Detections:", [
        (d["label"], round(d["confidence"], 3))
        for d in detections
    ])

    for frame in range(3):
        tracks = get_tracked_detections(image)
        print("Tracks", frame, [
            (d["label"], round(d["confidence"], 3), d["track_id"])
            for d in tracks
        ])

check("Before Phi")

print(run_phi_with_frame(image, "Describe this image briefly."))
check("After Phi, still loaded")

unload_phi()
check("After Phi unloaded")