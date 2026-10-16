# backend/rl/adaptive_policy.py
import json
import os

POLICY_PATH = "/home/unitree/go2-vlm-agent/models/adaptive_policy.json"

# default starting thresholds
DEFAULT_DEAD_ZONE = 60          # pixels either side of center before turning
DEFAULT_CLOSE_THRESHOLD = 600   # bounding box height to stop moving forward
DEFAULT_FORWARD_SPEED = 0.3     # m/s forward speed
DEFAULT_TURN_SPEED = 0.1        # rad/s turn speed

# adaptation limits — prevent thresholds drifting too far
MIN_DEAD_ZONE = 30
MAX_DEAD_ZONE = 200
MIN_CLOSE_THRESHOLD = 300
MAX_CLOSE_THRESHOLD = 900

# how aggressively thresholds adapt
ADAPTATION_RATE = 10            # pixels to adjust dead zone per oscillation event
DISTANCE_ADAPTATION_RATE = 30  # pixels to adjust close threshold per distance event

# how many consecutive events trigger an adaptation
OSCILLATION_TRIGGER = 5        # consecutive direction switches before widening dead zone
DISTANCE_TRIGGER = 5           # consecutive too-close/too-far events before adjusting threshold


class AdaptiveFollowPolicy:
    def __init__(self):
        self.dead_zone = DEFAULT_DEAD_ZONE
        self.close_threshold = DEFAULT_CLOSE_THRESHOLD
        self.forward_speed = DEFAULT_FORWARD_SPEED
        self.turn_speed = DEFAULT_TURN_SPEED

        # oscillation tracking
        self.last_action = None
        self.oscillation_count = 0

        # distance tracking
        self.too_close_count = 0
        self.too_far_count = 0

        # session stats for evaluation
        self.total_frames = 0
        self.centred_frames = 0
        self.target_lost_frames = 0
        self.adaptations = []

        self.load()

    def get_action(self, offset_x: float, box_height: float, target_visible: bool) -> str:
        """
        Select action based on current adaptive thresholds.
        Priority: safety > turn to centre > move forward > stop
        """
        self.total_frames += 1

        if not target_visible:
            self.target_lost_frames += 1
            return "search"

        # track centring accuracy for evaluation
        if abs(offset_x) <= self.dead_zone:
            self.centred_frames += 1

        # priority 1 — turn to centre target
        if offset_x > self.dead_zone:
            return "turn_right"
        elif offset_x < -self.dead_zone:
            return "turn_left"
        # priority 2 — move forward if far enough
        elif box_height < self.close_threshold:
            return "move_forward"
        # priority 3 — stop if close enough
        else:
            return "stop"

    def update(self, action: str, offset_x: float, box_height: float, target_visible: bool):
        """
        Observe the outcome of the last action and adapt thresholds if needed.
        Call this every frame after get_action().
        """
        if not target_visible:
            return

        self._check_oscillation(action)
        self._check_distance(box_height)

    def _check_oscillation(self, action: str):
        """
        Detect left-right oscillation and widen dead zone to dampen it.
        """
        if self.last_action in ("turn_left", "turn_right") and \
           action in ("turn_left", "turn_right") and \
           action != self.last_action:
            self.oscillation_count += 1
        else:
            self.oscillation_count = max(0, self.oscillation_count - 1)

        if self.oscillation_count >= OSCILLATION_TRIGGER:
            old = self.dead_zone
            self.dead_zone = min(self.dead_zone + ADAPTATION_RATE, MAX_DEAD_ZONE)
            self.oscillation_count = 0
            if self.dead_zone != old:
                self._log_adaptation(f"Oscillation detected — dead_zone {old} → {self.dead_zone}")

        self.last_action = action

    def _check_distance(self, box_height: float):
        """
        Detect consistent too-close or too-far following and adjust close threshold.
        """
        if box_height >= self.close_threshold:
            self.too_close_count += 1
            self.too_far_count = 0
        elif box_height < self.close_threshold * 0.3:
            self.too_far_count += 1
            self.too_close_count = 0
        else:
            self.too_close_count = max(0, self.too_close_count - 1)
            self.too_far_count = max(0, self.too_far_count - 1)

        # robot keeps stopping too early — lower the threshold
        if self.too_close_count >= DISTANCE_TRIGGER:
            old = self.close_threshold
            self.close_threshold = max(self.close_threshold - DISTANCE_ADAPTATION_RATE, MIN_CLOSE_THRESHOLD)
            self.too_close_count = 0
            if self.close_threshold != old:
                self._log_adaptation(f"Too close too often — close_threshold {old} → {self.close_threshold}")

        # robot never reaches target — raise the threshold
        if self.too_far_count >= DISTANCE_TRIGGER:
            old = self.close_threshold
            self.close_threshold = min(self.close_threshold + DISTANCE_ADAPTATION_RATE, MAX_CLOSE_THRESHOLD)
            self.too_far_count = 0
            if self.close_threshold != old:
                self._log_adaptation(f"Too far too often — close_threshold {old} → {self.close_threshold}")

    def _log_adaptation(self, message: str):
        print(f"[AdaptivePolicy] {message}")
        self.adaptations.append(message)

    def get_stats(self) -> dict:
        """
        Return session performance metrics for evaluation.
        """
        centring_accuracy = (self.centred_frames / self.total_frames * 100) if self.total_frames > 0 else 0
        target_retention = ((self.total_frames - self.target_lost_frames) / self.total_frames * 100) if self.total_frames > 0 else 0

        return {
            "total_frames": self.total_frames,
            "centring_accuracy_%": round(centring_accuracy, 1),
            "target_retention_%": round(target_retention, 1),
            "current_dead_zone": self.dead_zone,
            "current_close_threshold": self.close_threshold,
            "adaptations_made": len(self.adaptations),
            "adaptation_log": self.adaptations
        }

    def save(self):
        """Persist current thresholds to disk so they carry over between sessions."""
        state = {
            "dead_zone": self.dead_zone,
            "close_threshold": self.close_threshold,
            "forward_speed": self.forward_speed,
            "turn_speed": self.turn_speed
        }
        os.makedirs(os.path.dirname(POLICY_PATH), exist_ok=True)
        with open(POLICY_PATH, "w") as f:
            json.dump(state, f, indent=2)
        print(f"[AdaptivePolicy] Saved to {POLICY_PATH}")

    def load(self):
        """Load previously saved thresholds if available."""
        if os.path.exists(POLICY_PATH):
            with open(POLICY_PATH) as f:
                state = json.load(f)
            self.dead_zone = state.get("dead_zone", DEFAULT_DEAD_ZONE)
            self.close_threshold = state.get("close_threshold", DEFAULT_CLOSE_THRESHOLD)
            self.forward_speed = state.get("forward_speed", DEFAULT_FORWARD_SPEED)
            self.turn_speed = state.get("turn_speed", DEFAULT_TURN_SPEED)
            print(f"[AdaptivePolicy] Loaded — dead_zone={self.dead_zone}, close_threshold={self.close_threshold}")
        else:
            print("[AdaptivePolicy] No saved policy found — using defaults")

    def reset(self):
        """Reset thresholds back to defaults."""
        self.dead_zone = DEFAULT_DEAD_ZONE
        self.close_threshold = DEFAULT_CLOSE_THRESHOLD
        self.oscillation_count = 0
        self.too_close_count = 0
        self.too_far_count = 0
        self.adaptations = []
        print("[AdaptivePolicy] Reset to defaults")