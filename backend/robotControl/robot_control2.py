"""High-level Unitree Go2 motion helpers with lazy SDK initialization."""
from __future__ import annotations

import threading
import time

_action_lock = threading.RLock()
_client = None


def _get_client():
    """Create SportClient only when hardware control is actually requested."""
    global _client
    if _client is None:
        from unitree_sdk2py.go2.sport.sport_client import SportClient

        _client = SportClient()
        _client.SetTimeout(10.0)
        _client.Init()
    return _client


def execute_action(action: str):
    """Execute one named robot action; SDK errors are intentionally propagated."""
    with _action_lock:
        client = _get_client()
        if action == "move_forward": return client.Move(0.5, 0, 0)
        if action == "move_backward": return client.Move(-0.3, 0, 0)
        if action == "move_left": return client.Move(0, 0.3, 0)
        if action == "move_right": return client.Move(0, -0.3, 0)
        if action == "turn_left": return client.Move(0, 0, 0.5)
        if action == "turn_right": return client.Move(0, 0, -0.5)
        if action == "stop": return client.StopMove()

        if action == "stand_up": return client.StandUp()
        if action == "stand_down": return client.StandDown()
        if action == "balance_stand": return client.BalanceStand()
        if action == "sit": return client.Sit()
        if action == "rise_sit": return client.RiseSit()
        if action == "recovery_stand": return client.RecoveryStand()
        if action == "damp": return client.Damp()

        if action in ("hello", "emote_wave"): return client.Hello()
        if action in ("dance1", "emote_dance"): return client.Dance1()
        if action == "dance2": return client.Dance2()
        if action == "stretch": return client.Stretch()
        if action == "pose": return client.Pose(True)
        if action == "heart": return client.Heart()
        if action == "scrape": return client.Scrape()
        if action == "content": return client.Content()

        if action == "front_flip": return client.FrontFlip()
        if action == "front_jump": return client.FrontJump()
        if action == "front_pounce": return client.FrontPounce()
        if action == "left_flip": return client.LeftFlip()
        if action == "back_flip": return client.BackFlip()

        if action == "static_walk": return client.StaticWalk()
        if action == "trot_run": return client.TrotRun()
        if action == "free_walk": return client.FreeWalk()
        if action == "classic_walk_on": return client.ClassicWalk(True)
        if action == "classic_walk_off": return client.ClassicWalk(False)
        if action == "walk_upright_on": return client.WalkUpright(True)
        if action == "walk_upright_off": return client.WalkUpright(False)
        if action == "cross_step_on": return client.CrossStep(True)
        if action == "cross_step_off": return client.CrossStep(False)
        if action == "free_bound_on": return client.FreeBound(True)
        if action == "free_bound_off": return client.FreeBound(False)
        if action == "free_jump_on": return client.FreeJump(True)
        if action == "free_jump_off": return client.FreeJump(False)
        if action == "free_avoid_on": return client.FreeAvoid(True)
        if action == "free_avoid_off": return client.FreeAvoid(False)
        if action == "hand_stand_on": return client.HandStand(True)
        if action == "hand_stand_off": return client.HandStand(False)

        if action == "speed_slow": return client.SpeedLevel(1)
        if action == "speed_normal": return client.SpeedLevel(2)
        if action == "speed_fast": return client.SpeedLevel(3)

        raise ValueError(f"Unknown robot action: {action}")


def set_body_orientation(roll: float = 0.0, pitch: float = 0.0, yaw: float = 0.0):
    with _action_lock:
        return _get_client().Euler(float(roll), float(pitch), float(yaw))


def answer_gesture(detected: bool, amplitude: float = 0.20, pause: float = 0.22):
    """Yes = pitch nod; No = yaw shake using small whole-body Euler motions."""
    with _action_lock:
        client = _get_client()
        client.StopMove()
        client.BalanceStand()
        sequence = (
            [(0.0, amplitude, 0.0), (0.0, -amplitude, 0.0)]
            if detected
            else [(0.0, 0.0, amplitude), (0.0, 0.0, -amplitude)]
        )
        for _ in range(2):
            for roll, pitch, yaw in sequence:
                client.Euler(roll, pitch, yaw)
                time.sleep(pause)
        client.Euler(0.0, 0.0, 0.0)


def emergency_stop(damp: bool = True):
    """Best-effort emergency sequence: StopMove, then optionally Damp."""
    errors = []
    with _action_lock:
        client = _get_client()
        try:
            client.StopMove()
        except Exception as exc:
            errors.append(exc)
        if damp:
            try:
                client.Damp()
            except Exception as exc:
                errors.append(exc)
    if errors:
        raise RuntimeError("Emergency stop had SDK errors") from errors[0]