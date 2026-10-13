import unittest
from backend.target_observer import parse_target, TargetObserver

class TestTargetObserver(unittest.TestCase):
    def test_command(self):
        self.assertEqual(parse_target('follow the blue bottle'), ('blue', 'bottle'))
    def test_accuracy_empty(self):
        self.assertIsNone(TargetObserver().accuracy)
    def test_accuracy(self):
        observer = TargetObserver()
        observer.total_visible_frames = 10
        observer.centred_frames = 7
        self.assertEqual(observer.accuracy, 70.0)

if __name__ == '__main__':
    unittest.main()
