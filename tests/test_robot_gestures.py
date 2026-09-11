import backend.robotControl.robot_control as rc


class FakeClient:
    def __init__(self): self.calls = []
    def StopMove(self): self.calls.append(("stop",))
    def BalanceStand(self): self.calls.append(("balance",))
    def Euler(self, r, p, y): self.calls.append(("euler", r, p, y))


def test_yes_gesture_uses_pitch_and_returns_neutral(monkeypatch):
    fake = FakeClient()
    monkeypatch.setattr(rc, "_client", fake)
    monkeypatch.setattr(rc.time, "sleep", lambda _: None)
    rc.answer_gesture(True)
    eulers = [c for c in fake.calls if c[0] == "euler"]
    assert any(abs(c[2]) > 0 and c[3] == 0 for c in eulers[:-1])
    assert eulers[-1] == ("euler", 0.0, 0.0, 0.0)


def test_no_gesture_uses_yaw_and_returns_neutral(monkeypatch):
    fake = FakeClient()
    monkeypatch.setattr(rc, "_client", fake)
    monkeypatch.setattr(rc.time, "sleep", lambda _: None)
    rc.answer_gesture(False)
    eulers = [c for c in fake.calls if c[0] == "euler"]
    assert any(abs(c[3]) > 0 and c[2] == 0 for c in eulers[:-1])
    assert eulers[-1] == ("euler", 0.0, 0.0, 0.0)
