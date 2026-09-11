import backend.decision_logic as dl


class ReadyCamera:
    def is_ready(self):
        return True


def test_detect_yes_triggers_yes_gesture(monkeypatch):
    gestures = []
    monkeypatch.setattr(dl, "detect_object", lambda target, camera: True)
    monkeypatch.setattr(dl, "answer_gesture", gestures.append)
    dl.run_task({"action": "detect", "target": "bottle"}, ReadyCamera())
    assert gestures == [True]


def test_detect_no_triggers_no_gesture(monkeypatch):
    gestures = []
    monkeypatch.setattr(dl, "detect_object", lambda target, camera: False)
    monkeypatch.setattr(dl, "answer_gesture", gestures.append)
    dl.run_task({"action": "see", "target": "chair"}, ReadyCamera())
    assert gestures == [False]


def test_unknown_action_stops(monkeypatch):
    actions = []
    monkeypatch.setattr(dl, "execute_action", actions.append)
    dl.run_task({"action": "nonsense", "target": None}, ReadyCamera())
    assert actions == ["stop"]


def test_prompt_to_detection_gesture_pipeline(monkeypatch):
    from backend.utils.input_processor import process_input

    gestures = []
    task = process_input("Can you see a bottle?")
    monkeypatch.setattr(dl, "detect_object", lambda target, camera: target == "bottle")
    monkeypatch.setattr(dl, "answer_gesture", gestures.append)

    dl.run_task(task, ReadyCamera())
    assert task == {"action": "see", "target": "bottle"}
    assert gestures == [True]
