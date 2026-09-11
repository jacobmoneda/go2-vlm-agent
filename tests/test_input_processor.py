from backend.utils.input_processor import process_input


def test_question_parses_detectable_object():
    assert process_input("Can you see a bottle?") == {"action": "see", "target": "bottle"}


def test_follow_defaults_to_person_when_named():
    assert process_input("Please follow the person") == {"action": "follow", "target": "person"}
