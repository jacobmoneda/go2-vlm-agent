import io
from PIL import Image

from backend.autonomy import PalmPostureController
from backend.safety import KillSwitch


def jpeg_bytes():
    image = Image.new("RGB", (8, 8))
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    return buf.getvalue()


class Camera:
    def is_ready(self):
        return True

    def get_frame_bytes(self):
        return jpeg_bytes()


def test_palm_sits_and_absence_stands_with_debounce():
    actions = []
    states = iter([True, True, False, False])
    detector = lambda _img: next(states)
    ks = KillSwitch(lambda: None)
    controller = PalmPostureController(
        Camera(), actions.append, ks, detector=detector, required_frames=2
    )

    assert controller.step() is None
    assert controller.step() == "sit"
    assert controller.step() is None
    assert controller.step() == "stand"
    assert actions == ["sit", "stand_up"]


def test_palm_controller_does_nothing_when_killed():
    actions = []
    ks = KillSwitch(lambda: None)
    ks.engage("test")
    controller = PalmPostureController(Camera(), actions.append, ks, detector=lambda _: True)
    assert controller.step() is None
    assert actions == []
