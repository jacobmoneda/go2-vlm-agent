# Safe target tracking evaluation

This addition performs camera-only observation and centring measurement. It never sends motion commands.

From the repository root, with existing robot SDK, ultralytics, OpenCV and Pillow installed, run:

```python
from backend.camera.go2_camera import Go2Camera
from backend.target_observer import TargetObserver
import time

camera = Go2Camera(network_interface='eth0')
observer = TargetObserver(dead_zone_fraction=0.10)
camera.start()
try:
    while True:
        frame = camera.get_frame()
        if frame is not None:
            print(observer.observe(frame, 'track the blue bottle'))
        time.sleep(0.2)
finally:
    camera.stop()
```

The centring accuracy is `100 * centred_frames / visible_target_frames`. Here the dead zone is +/-10% of the half-frame width. This measures target centring only, not recognition accuracy, navigation success, or physical distance. Missed-target frames are excluded from this denominator and should be reported separately. COCO YOLO may detect 'bottle' but cannot distinguish all arbitrary objects; the HSV colour test is only a heuristic. No robot movement is enabled. Keep a human supervisor and manufacturer-approved collision protection for any subsequent physical trials.
