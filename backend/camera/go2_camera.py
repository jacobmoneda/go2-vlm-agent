# backend/camera/go2_camera.py

import threading
import time
from typing import Optional

import numpy as np
import cv2
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.video.video_client import VideoClient


class Go2Camera:
    """
    Continuous camera stream from Unitree Go2 via VideoClient SDK.
    Runs capture loop in background thread.
    Exposes get_frame() matching the shared camera abstraction.
    """

    def __init__(self, network_interface: str = None):
        if network_interface:
            ChannelFactoryInitialize(0, network_interface)
        else:
            ChannelFactoryInitialize(0)

        self.client = VideoClient()
        self.client.SetTimeout(3.0)
        self.client.Init()

        self._frame_raw: bytes = None        # latest raw JPEG bytes
        self._frame_np: np.ndarray = None    # latest decoded numpy array
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

        self.fps_target = 10  # Go2 camera, 10fps is safe; raise if stable
        self._frame_interval = 1.0 / self.fps_target

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        self._running = True
        self._thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,
            name="Go2CameraThread"
        )
        self._thread.start()
        print("[Go2Camera] Stream started.")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
        print("[Go2Camera] Stream stopped.")

    # ------------------------------------------------------------------
    # Capture loop (background thread)
    # ------------------------------------------------------------------

    def _capture_loop(self):
        while self._running:
            t0 = time.time()

            code, data = self.client.GetImageSample()

            if code == 0 and data:
                raw_bytes = bytes(data)
                np_arr = np.frombuffer(raw_bytes, dtype=np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                if frame is not None:
                    with self._lock:
                        self._frame_raw = raw_bytes
                        self._frame_np = frame
            else:
                print(f"[Go2Camera] GetImageSample error. code={code}")

            # Throttle to fps_target
            elapsed = time.time() - t0
            sleep_time = self._frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    # ------------------------------------------------------------------
    # Public interface — matches WebcamCamera abstraction
    # ------------------------------------------------------------------

    def get_frame(self) -> Optional[np.ndarray]:
        """Returns latest frame as BGR numpy array. Returns None if no frame yet."""
        with self._lock:
            return self._frame_np.copy() if self._frame_np is not None else None

    def get_frame_bytes(self) -> Optional[bytes]:
        """Returns latest frame as raw JPEG bytes — direct input for Qwen-VL."""
        with self._lock:
            return self._frame_raw

    def is_ready(self) -> bool:
        """True once at least one frame has been received."""
        with self._lock:
            return self._frame_np is not None