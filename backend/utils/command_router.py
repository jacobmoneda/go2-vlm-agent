# backend/utils/command_router.py
import json
import re
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "tinyllama"  # swap to "phi3-mini" if installed

SYSTEM_PROMPT = """You are a command parser for a quadruped robot dog.
Given a user command, determine whether it requires visual context and what action to take.

Valid actions: move_forward, move_backward, move_left, move_right, turn_left,
turn_right, stop, sit, stand_up, stand_down, hello, dance1, dance2, stretch,
pose, heart, front_flip, back_flip, trot_run, speed_slow, speed_normal,
speed_fast, search, walk_upright_on, classic_walk_on

Rules:
- needs_vision=true: commands that require seeing the camera feed to decide
- needs_vision=false: commands that can be executed without seeing anything
- is_follow_command=true: any follow/track/chase/find command — YOLO handles these

Respond ONLY with a JSON object. No explanation, no markdown, no extra text.
Format: {"needs_vision": bool, "action": "action_name or null", "is_follow_command": bool, "confidence": float, "reasoning": "one sentence"}

Examples:
"sit down" -> {"needs_vision": false, "action": "sit", "is_follow_command": false, "confidence": 0.99, "reasoning": "direct sit command requires no vision"}
"wave hello" -> {"needs_vision": false, "action": "hello", "is_follow_command": false, "confidence": 0.99, "reasoning": "direct emote command"}
"follow me" -> {"needs_vision": false, "action": null, "is_follow_command": true, "confidence": 0.99, "reasoning": "follow command routed to YOLO"}
"stop when you see a chair" -> {"needs_vision": true, "action": null, "is_follow_command": false, "confidence": 0.9, "reasoning": "requires visual context to detect chair"}
"do a backflip" -> {"needs_vision": false, "action": "back_flip", "is_follow_command": false, "confidence": 0.99, "reasoning": "direct acrobatic command"}
"go faster" -> {"needs_vision": false, "action": "speed_fast", "is_follow_command": false, "confidence": 0.95, "reasoning": "speed adjustment command"}
"what do you see" -> {"needs_vision": true, "action": null, "is_follow_command": false, "confidence": 0.99, "reasoning": "descriptive command requires vision"}"""


def parse_command(user_command: str) -> dict:
    """
    Send user command to local Ollama LLM and parse the JSON response.
    Returns a dict with needs_vision, action, is_follow_command, confidence, reasoning.
    """
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": f"{SYSTEM_PROMPT}\n\nCommand: {user_command}",
        "stream": False,
        "options": {
            "temperature": 0.1,   # low temperature for consistent output
            "num_predict": 150    # limit tokens — we only need a small JSON
        }
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=10)
        response.raise_for_status()
        raw = response.json().get("response", "")
        print(f"[Router] Raw LLM output: {raw}")
        return _parse_json(raw)

    except requests.exceptions.ConnectionError:
        print("[Router] Ollama not running — falling back to keyword routing")
        return _keyword_fallback(user_command)

    except requests.exceptions.Timeout:
        print("[Router] Ollama timeout — falling back to keyword routing")
        return _keyword_fallback(user_command)

    except Exception as e:
        print(f"[Router] Error: {e} — falling back to keyword routing")
        return _keyword_fallback(user_command)


def _parse_json(raw: str) -> dict:
    """Extract JSON from LLM output with fallback."""
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    print("[Router] JSON parse failed — defaulting to stop")
    return _default_response()


def _keyword_fallback(user_command: str) -> dict:
    """
    Simple keyword-based fallback if Ollama is unavailable.
    Mirrors the original routing logic in main.py.
    """
    cmd = user_command.lower()

    # follow commands
    if any(kw in cmd for kw in ["follow", "track", "chase", "find", "locate"]):
        return {"needs_vision": False, "action": None, "is_follow_command": True, "confidence": 0.9, "reasoning": "keyword match: follow command"}

    # direct action commands
    if any(kw in cmd for kw in ["sit", "sit down"]):
        return {"needs_vision": False, "action": "sit", "is_follow_command": False, "confidence": 0.9, "reasoning": "keyword match: sit"}
    if any(kw in cmd for kw in ["stand up", "standup", "get up"]):
        return {"needs_vision": False, "action": "stand_up", "is_follow_command": False, "confidence": 0.9, "reasoning": "keyword match: stand up"}
    if any(kw in cmd for kw in ["wave", "hello", "hi"]):
        return {"needs_vision": False, "action": "hello", "is_follow_command": False, "confidence": 0.9, "reasoning": "keyword match: wave"}
    if any(kw in cmd for kw in ["dance", "dance1"]):
        return {"needs_vision": False, "action": "dance1", "is_follow_command": False, "confidence": 0.9, "reasoning": "keyword match: dance"}
    if any(kw in cmd for kw in ["stop", "halt", "freeze"]):
        return {"needs_vision": False, "action": "stop", "is_follow_command": False, "confidence": 0.9, "reasoning": "keyword match: stop"}
    if any(kw in cmd for kw in ["faster", "speed up"]):
        return {"needs_vision": False, "action": "speed_fast", "is_follow_command": False, "confidence": 0.9, "reasoning": "keyword match: speed up"}
    if any(kw in cmd for kw in ["slower", "slow down"]):
        return {"needs_vision": False, "action": "speed_slow", "is_follow_command": False, "confidence": 0.9, "reasoning": "keyword match: slow down"}
    if any(kw in cmd for kw in ["backflip", "back flip"]):
        return {"needs_vision": False, "action": "back_flip", "is_follow_command": False, "confidence": 0.9, "reasoning": "keyword match: backflip"}
    if any(kw in cmd for kw in ["forward", "move forward"]):
        return {"needs_vision": False, "action": "move_forward", "is_follow_command": False, "confidence": 0.9, "reasoning": "keyword match: forward"}
    if any(kw in cmd for kw in ["back", "backward", "move back"]):
        return {"needs_vision": False, "action": "move_backward", "is_follow_command": False, "confidence": 0.9, "reasoning": "keyword match: backward"}

    # vision-dependent fallback
    return {"needs_vision": True, "action": None, "is_follow_command": False, "confidence": 0.5, "reasoning": "unknown command — routing to vision"}


def _default_response() -> dict:
    return {
        "needs_vision": False,
        "action": "stop",
        "is_follow_command": False,
        "confidence": 0.0,
        "reasoning": "parse failure — defaulting to stop"
    }