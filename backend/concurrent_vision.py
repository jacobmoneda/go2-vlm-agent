"""Continuous YOLO tracking with asynchronous, slower VLM descriptions.
Observation-only: never sends motor or navigation commands.
"""
import argparse
import json
import threading
import time
import cv2
from PIL import Image
from backend.camera.go2_camera import Go2Camera
from backend.target_observer import TargetObserver, parse_target
from backend.vlm.qwen_engine import QwenObserver

class ConcurrentVision:
    def __init__(self, command, vlm_interval=8.0, dead_zone=0.10, enable_vlm=True):
        parse_target(command)
        self.command = command
        self.observer = TargetObserver(dead_zone_fraction=dead_zone)
        self.vlm = QwenObserver() if enable_vlm else None
        self.vlm_interval = max(1.0, vlm_interval)
        self.last_vlm = None
        self.last_vlm_error = None
        self._busy = False
        self._lock = threading.Lock()
        self._last_submit = 0.0

    def _vlm_job(self, frame):
        try:
            rgb = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = self.vlm.describe(rgb, self.command)
            with self._lock:
                self.last_vlm, self.last_vlm_error = result, None
        except Exception as exc:
            with self._lock:
                self.last_vlm_error = f'{type(exc).__name__}: {exc}'
        finally:
            with self._lock:
                self._busy = False

    def process(self, frame):
        result = self.observer.observe(frame, self.command)
        if self.vlm is not None:
            with self._lock:
                if not self._busy and time.monotonic() - self._last_submit >= self.vlm_interval:
                    self._busy = True
                    self._last_submit = time.monotonic()
                    threading.Thread(target=self._vlm_job, args=(frame.copy(),), daemon=True).start()
                result['vlm_description'] = self.last_vlm
                result['vlm_error'] = self.last_vlm_error
                result['vlm_processing'] = self._busy
        result['movement_executed'] = False
        result['movement_decision'] = ('hold: target not detected' if not result['visible'] else
            'hold: target centred' if abs(result['horizontal_error_percent']) <= self.observer.dead_zone_fraction * 100 else
            'suggest turn left (not executed)' if result['horizontal_error_percent'] < 0 else
            'suggest turn right (not executed)')
        return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--interface', default='eth0')
    parser.add_argument('--command', default='track the blue bottle')
    parser.add_argument('--vlm-interval', type=float, default=8)
    parser.add_argument('--dead-zone', type=float, default=0.10)
    parser.add_argument('--no-vlm', action='store_true')
    args = parser.parse_args()
    camera = Go2Camera(args.interface)
    pipeline = ConcurrentVision(args.command, args.vlm_interval, args.dead_zone, not args.no_vlm)
    camera.start()
    try:
        while True:
            frame = camera.get_frame()
            if frame is not None:
                print(json.dumps(pipeline.process(frame), ensure_ascii=False), flush=True)
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        camera.stop()

if __name__ == '__main__':
    main()
