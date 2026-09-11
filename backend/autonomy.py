"""Continuous safety behaviours that have priority over user tasks."""
from __future__ import annotations

import io
import threading
import time
from PIL import Image

from backend.objectDetection.palm_engine import detect_open_palm


class PalmPostureController:
    """Sit on open palm, stand when it disappears, with temporal debounce."""

    def __init__(
        self,
        camera,
        execute_action,
        killswitch,
        detector=detect_open_palm,
        interval: float = 0.15,
        required_frames: int = 2,
    ):
        self.camera = camera
        self.execute_action = execute_action
        self.killswitch = killswitch
        self.detector = detector
        self.interval = interval
        self.required_frames = max(1, required_frames)
        self._last_raw = None
        self._streak = 0
        self._posture = None
        self.palm_active = False
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def step(self) -> str | None:
        """Process one frame and return posture transition, if any."""
        if self.killswitch.engaged or not self.camera.is_ready():
            return None

        frame_bytes = self.camera.get_frame_bytes()
        if not frame_bytes:
            return None

        image = Image.open(io.BytesIO(frame_bytes)).convert("RGB")
        palm_seen = bool(self.detector(image))

        if palm_seen == self._last_raw:
            self._streak += 1
        else:
            self._last_raw = palm_seen
            self._streak = 1

        if self._streak < self.required_frames:
            return None

        desired = "sit" if palm_seen else "stand"
        self.palm_active = palm_seen
        if desired == self._posture:
            return None

        self.execute_action("sit" if palm_seen else "stand_up")
        self._posture = desired
        print(f"[Palm] open_palm={palm_seen} -> {desired}")
        return desired

    def run(self):
        print("[Palm] Continuous palm posture controller started.")
        while not self._stop_event.is_set():
            try:
                self.step()
            except FileNotFoundError as exc:
                # Missing mandatory perception weights is a configuration fault.
                self.killswitch.engage(str(exc), source="palm_detector")
                return
            except Exception as exc:
                self.killswitch.engage(repr(exc), source="palm_controller")
                return
            time.sleep(self.interval)
