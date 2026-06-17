import sys
import time
import cv2
from PIL import Image

from backend.camera.go2_camera import Go2Camera
from backend.vlm.smolvlm_engine import run_smolvlm_with_frame


ROBOT_PROMPT = """
You are the vision system of a robot dog.

Look at the image and return only these three lines:

Type: choose one from human, animal, or object.
Hand gesture: describe only the visible hand/finger gesture.
Location: choose indoor or outdoor.

Do NOT describe clothing, glasses, face details, room details, ceiling, lights, or background objects.
Do NOT describe the person's intention or activity, such as taking a selfie.
Only describe the visible hand gesture.

Use this exact format:
Type:
Hand gesture:
Location:
"""

SHOW_CAMERA_WINDOW = True
SAVE_PNG = True
PNG_FILENAME = "latest_go2_smolvlm_frame.png"
IMAGE_SIZE = (224, 224)


def bgr_to_pil(frame_bgr):
    """
    Convert OpenCV BGR frame to PIL RGB image.
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb).convert("RGB")
    image = image.resize(IMAGE_SIZE)

    if SAVE_PNG:
        image.save(PNG_FILENAME, format="PNG")

    return image


def main():
    if len(sys.argv) >= 2:
        interface = sys.argv[1]
        cam = Go2Camera(network_interface=interface)
        print(f"[Main] Using network interface: {interface}")
    else:
        cam = Go2Camera()
        print("[Main] Using default network interface")

    cam.start()

    print("[Main] Waiting for first Go2 frame...")

    while not cam.is_ready():
        time.sleep(0.1)

    print("[Main] First frame received.")
    print("[Main] Starting SmolVLM2 inference loop. Press Ctrl+C to stop.")

    try:
        while True:
            frame = cam.get_frame()

            if frame is None:
                print("[Main] No frame available yet.")
                time.sleep(0.1)
                continue

            if SHOW_CAMERA_WINDOW:
                preview = cv2.resize(frame, (320, 180))
                cv2.imshow("Go2 Camera Window", preview)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Stopped by q key.")
                    break

            pil_image = bgr_to_pil(frame)

            result = run_smolvlm_with_frame(pil_image, ROBOT_PROMPT)

            print("\n=== SmolVLM2 ROBOT VISION OUTPUT ===")
            print(result)

            if SAVE_PNG:
                print(f"[PNG] Saved latest frame as: {PNG_FILENAME}")

            time.sleep(1)

    except KeyboardInterrupt:
        print("Stopped stream.")

    finally:
        cam.stop()
        cv2.destroyAllWindows()
        print("Camera stopped.")


if __name__ == "__main__":
    main()