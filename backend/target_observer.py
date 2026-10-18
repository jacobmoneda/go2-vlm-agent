"""Camera-only target observer. Does not send robot motion commands."""
import re
import time
from collections import Counter
from PIL import Image
from backend.objectDetection.yolo_engine import get_tracked_detections

COLORS = {'blue', 'red', 'green', 'yellow', 'white', 'black', 'orange', 'purple'}


def parse_target(command):
    match = re.fullmatch(r'\s*(?:follow|track|find)\s+(?:the\s+)?(?:(blue|red|green|yellow|white|black|orange|purple)\s+)?([\w -]+?)\s*', command.lower())
    if not match:
        raise ValueError('Use e.g. track the blue bottle')
    return match.group(1), match.group(2).strip()


def color_matches(rgb_crop, color):
    """Simple colour heuristic, not a calibrated classifier."""
    if not color:
        return True
    import cv2
    import numpy as np
    hsv = cv2.cvtColor(np.asarray(rgb_crop), cv2.COLOR_RGB2HSV)
    ranges = {
        'blue': ((90, 55, 45), (135, 255, 255)),
        'green': ((35, 50, 45), (89, 255, 255)),
        'yellow': ((18, 70, 60), (35, 255, 255)),
        'red': ((0, 80, 60), (10, 255, 255)),
        'orange': ((10, 80, 60), (18, 255, 255)),
        'purple': ((136, 55, 45), (165, 255, 255)),
        'white': ((0, 0, 175), (179, 55, 255)),
        'black': ((0, 0, 0), (179, 255, 65)),
    }
    lo, hi = ranges[color]
    mask = cv2.inRange(hsv, np.array(lo), np.array(hi))
    if color == 'red':
        mask |= cv2.inRange(hsv, np.array((170,80,60)), np.array((179,255,255)))
    return float((mask > 0).mean()) >= 0.12


class TargetObserver:
    def __init__(self, dead_zone_fraction=0.10):
        self.dead_zone_fraction = dead_zone_fraction
        self.total_visible_frames = 0
        self.centred_frames = 0
        self.last_track_id = None

    def reset(self):
        self.total_visible_frames = 0
        self.centred_frames = 0
        self.last_track_id = None

    def observe(self, frame_bgr, command):
        import cv2
        color, noun = parse_target(command)
        image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        width, height = image.size
        candidates = []
        for det in get_tracked_detections(image):
            label = det['label'].lower()
            # Standard COCO YOLO labels 'bottle', but not arbitrary open vocabulary.
            if label != noun and not (noun.endswith('s') and label == noun[:-1]):
                continue
            if det['confidence'] < 0.40:
                continue
            x1, y1, x2, y2 = [int(v) for v in det['xyxy']]
            crop = image.crop((max(0,x1), max(0,y1), min(width,x2), min(height,y2)))
            if crop.width < 2 or crop.height < 2 or not color_matches(crop, color):
                continue
            candidates.append(det)
        if not candidates:
            self.last_track_id = None
            return {'target': f'{color or ""} {noun}'.strip(), 'visible': False,
                    'observation': 'target not confidently detected', 'action': 'none',
                    'centring_accuracy_percent': self.accuracy}
        candidates.sort(key=lambda d: (d.get('track_id') == self.last_track_id, d['confidence']), reverse=True)
        det = candidates[0]
        self.last_track_id = det.get('track_id')
        error = (det['box_center_x'] - width / 2) / (width / 2)
        centred = abs(error) <= self.dead_zone_fraction
        self.total_visible_frames += 1
        self.centred_frames += int(centred)
        position = 'centre' if centred else ('left' if error < 0 else 'right')
        return {'target': f'{color or ""} {noun}'.strip(), 'visible': True,
                'confidence': round(det['confidence'], 3), 'track_id': self.last_track_id,
                'relative_box_height_percent': round(100 * (det['xyxy'][3] - det['xyxy'][1]) / height, 2),
                'horizontal_error_percent': round(error * 100, 1),
                'observation': f'target {position}', 'action': 'none (observation-only)',
                'centring_accuracy_percent': self.accuracy,
                'visible_frames': self.total_visible_frames, 'centred_frames': self.centred_frames}

    @property
    def accuracy(self):
        if not self.total_visible_frames:
            return None
        return round(100 * self.centred_frames / self.total_visible_frames, 2)
