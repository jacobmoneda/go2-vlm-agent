"""Open-palm detector with lazy model loading.

Set PALM_MODEL_PATH to the trained hand/palm YOLO weights on the robot.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

DEFAULT_PALM_MODEL_PATH = os.environ.get(
    "PALM_MODEL_PATH", "/home/unitree/models/palm_detector.pt"
)
PALM_CONFIDENCE_THRESHOLD = float(os.environ.get("PALM_CONFIDENCE_THRESHOLD", "0.60"))
PALM_LABELS = {
    label.strip().lower()
    for label in os.environ.get("PALM_LABELS", "palm,open_palm,open palm,hand").split(",")
    if label.strip()
}

_model = None
_model_path: Optional[str] = None


def _load_model(model_path: str = DEFAULT_PALM_MODEL_PATH):
    global _model, _model_path
    if _model is not None and _model_path == model_path:
        return _model

    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Palm model not found at {model_path}. Set PALM_MODEL_PATH to your trained weights."
        )

    from ultralytics import YOLO

    _model = YOLO(model_path)
    _model_path = model_path
    return _model


def detect_open_palm(
    pil_image: Image.Image,
    confidence_threshold: float = PALM_CONFIDENCE_THRESHOLD,
    model_path: str = DEFAULT_PALM_MODEL_PATH,
) -> bool:
    """Return True when the configured YOLO model sees an open palm."""
    model = _load_model(model_path)
    results = model(np.array(pil_image), verbose=False)

    for box in results[0].boxes:
        confidence = float(box.conf)
        if confidence < confidence_threshold:
            continue
        label = str(model.names[int(box.cls)]).strip().lower()
        if label in PALM_LABELS:
            return True
    return False