import sys
import threading
import queue
import cv2
import numpy as np
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_go.msg.dds_ import Go2FrontVideoData_

class Go2CameraStreamer:
    def __init__(self, interface: str):
        self.interface = interface
        self.frame_queue = queue.Queue(maxsize=2)  # Low maxsize keeps latency down
        self.is_running = False
        self.sub = None

    def _video_callback(self, msg: Go2FrontVideoData_):
        if msg.video720p and len(msg.video720p) > 0:
            np_arr = np.frombuffer(bytes(msg.video720p), dtype=np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if frame is not None:
                # If queue is full, drop the oldest frame to prevent VLM lag
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.frame_queue.put_nowait(frame)

    def start(self):
        """Starts the DDS subscription loop in the background."""
        ChannelFactoryInitialize(0, self.interface)
        self.sub = ChannelSubscriber("rt/frontvideostream", Go2FrontVideoData_)
        self.sub.Init(self._video_callback, 10)
        self.is_running = True
        print(f"[SDK] Subscribed to camera stream on {self.interface}")

    def get_latest_frame(self, timeout=None):
        """Call this from your VLM loop to pull the freshest image array."""
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None
        

if __name__ == "__main__":
    interface = sys.argv[1] if len(sys.argv) > 1 else "eth0"
    
    streamer = Go2CameraStreamer(interface)
    streamer.start()

    print("Press 'q' to quit.")

    while True:
        frame = streamer.get_latest_frame(timeout=1.0)

        if frame is None:
            print("Waiting for frame...")
            continue

        cv2.imshow("Go2 Camera", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()