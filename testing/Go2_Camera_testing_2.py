# Go2_Camera_testing_2.py
# Run from project root:
# python .\Go2_Camera_testing_2.py

import time
import sys
from pathlib import Path

# Make sure project root is in Python path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.camera.go2_camera import Go2Camera


def main():
    print("Starting Unitree Go2 camera test...")

    # Default camera setup
    # If this does not work, try:
    # cam = Go2Camera("eth0")
    # cam = Go2Camera("Ethernet")
    cam = Go2Camera()

    try:
        cam.start()

        print("Waiting for first frame...")

        timeout_seconds = 20
        start_time = time.time()

        while not cam.is_ready():
            if time.time() - start_time > timeout_seconds:
                print("ERROR: No camera frame received within 20 seconds.")
                print("Check:")
                print("1. Go2 robot is turned on")
                print("2. Computer is connected to Go2 network")
                print("3. Correct network interface is used")
                print("4. Go2 camera stream is active")
                return

            time.sleep(0.1)

        print("First frame received.")

        # Grab 5 frames and save each one
        for i in range(5):
            raw = cam.get_frame_bytes()

            if raw:
                filename = PROJECT_ROOT / f"test_frame_{i}.jpg"

                with open(filename, "wb") as f:
                    f.write(raw)

                print(f"Saved {filename}")
            else:
                print(f"No frame received for frame {i}")

            time.sleep(0.5)

        print("Camera test finished successfully.")

    except KeyboardInterrupt:
        print("Stopped by user.")

    except Exception as e:
        print("ERROR:", e)

    finally:
        try:
            cam.stop()
            print("Camera stopped.")
        except Exception:
            pass


if __name__ == "__main__":
    main()