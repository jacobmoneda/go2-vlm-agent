# backend/utils/command_router.py
import json
import re
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "phi3-fast"
SYSTEM_PROMPT = """You are a command router for a Unitree Go2 robot.

Your only job is to classify the user's command and return exactly one JSON object.
Do not output markdown, explanations, code fences, or text outside the JSON.

AVAILABLE ROBOT ACTIONS:
move_forward
move_backward
move_left
move_right
turn_left
turn_right
stop
sit
stand_up
stand_down
hello
dance1
dance2
stretch
pose
heart
front_flip
back_flip
trot_run
speed_slow
speed_normal
speed_fast
search

OUTPUT FORMAT:
{"needs_vision": false, "action": null, "is_follow_command": false, "confidence": 0.0, "reasoning": "short reason"}

CLASSIFICATION RULES:

1. DIRECT ROBOT COMMAND
Use this when the robot can perform the command without looking at the camera.

Set:
- needs_vision = false
- is_follow_command = false
- action = the matching robot action

Examples:
"move forward"
-> {"needs_vision": false, "action": "move_forward", "is_follow_command": false, "confidence": 1.0, "reasoning": "direct movement command"}

"can you wave?"
-> {"needs_vision": false, "action": "hello", "is_follow_command": false, "confidence": 1.0, "reasoning": "wave maps to hello"}

"sit down"
-> {"needs_vision": false, "action": "sit", "is_follow_command": false, "confidence": 1.0, "reasoning": "direct posture command"}


2. FOLLOW OR TRACK COMMAND
Use this when the user asks the robot to follow, track, chase, or continuously watch a visible target.

Set:
- is_follow_command = true
- needs_vision = true
- action = null

Examples:
"follow the person"
-> {"needs_vision": true, "action": null, "is_follow_command": true, "confidence": 1.0, "reasoning": "following requires target detection"}

"track the bottle"
-> {"needs_vision": true, "action": null, "is_follow_command": true, "confidence": 1.0, "reasoning": "tracking requires target detection"}


3. VISUAL COMMAND
Use this when answering or performing the command requires information from the camera, but the command is not a follow/track command.

Examples include:
- asking whether an object is visible
- asking what the robot can see
- identifying or describing an object or person
- locating an object

Set:
- needs_vision = true
- is_follow_command = false
- action = null

Examples:
"do you see a bottle?"
-> {"needs_vision": true, "action": null, "is_follow_command": false, "confidence": 1.0, "reasoning": "requires checking camera for bottle"}

"what can you see?"
-> {"needs_vision": true, "action": null, "is_follow_command": false, "confidence": 1.0, "reasoning": "requires visual perception"}

"where is the chair?"
-> {"needs_vision": true, "action": null, "is_follow_command": false, "confidence": 1.0, "reasoning": "requires locating an object"}


4. NATURAL LANGUAGE MAPPINGS
Interpret different wording with the same meaning as the corresponding action:

wave, say hello, greet -> hello
dance -> dance1
backflip, back flip -> back_flip
frontflip, front flip -> front_flip
forward, go forward -> move_forward
backward, go back, move back -> move_backward
move left, step left -> move_left
move right, step right -> move_right
turn left -> turn_left
turn right -> turn_right
faster, speed up -> speed_fast
slower, slow down -> speed_slow
normal speed -> speed_normal
halt, freeze -> stop
get up -> stand_up


5. IMPORTANT DISTINCTIONS

"move left"
-> move_left

"turn left"
-> turn_left

"look left"
-> requires vision; do not treat it as move_left

"follow the person"
-> follow command; do not return move_forward

"find the bottle"
-> requires vision; do not return a direct movement action

"stop following"
-> stop

Do not invent actions that are not in AVAILABLE ROBOT ACTIONS.

If the command is ambiguous or cannot be confidently mapped to a direct robot action, do not guess. Route it to vision:
{"needs_vision": true, "action": null, "is_follow_command": false, "confidence": 0.5, "reasoning": "command requires further interpretation"}

Command: """

# normalise action names that models commonly return incorrectly
ACTION_ALIASES = {
    "dance":      "dance1",
    "wave":       "hello",
    "wave_hello": "hello",
    "emote_wave": "hello",
    "emote_sit":  "sit",
    "emote_dance":"dance1",
    "backflip":   "back_flip",
    "frontflip":  "front_flip",
    "faster":     "speed_fast",
    "slower":     "speed_slow",
    "forward":    "move_forward",
    "backward":   "move_backward",
    "back":       "move_backward",
}

VALID_ACTIONS = {
    "move_forward", "move_backward", "move_left", "move_right",
    "turn_left", "turn_right", "stop", "sit", "stand_up",
    "stand_down", "hello", "dance1", "dance2", "stretch", "pose",
    "heart", "front_flip", "back_flip", "trot_run", "speed_slow",
    "speed_normal", "speed_fast", "search", "walk_upright_on",
    "classic_walk_on"
}


def strip_markdown(raw: str) -> str:
    """Remove markdown code fences if present."""
    raw = raw.strip()
    raw = re.sub(r'^```json\s*', '', raw)
    raw = re.sub(r'^```\s*', '', raw)
    raw = re.sub(r'\s*```$', '', raw)
    return raw.strip()


def extract_json(raw: str) -> dict:
    """Try multiple strategies to extract JSON from raw output."""
    # strategy 1 — direct parse after stripping markdown
    cleaned = strip_markdown(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # strategy 2 — find first { ... } block
    match = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # strategy 3 — extract individual fields via regex (handles truncated output)
    result = {}
    nv = re.search(r'"needs_vision"\s*:\s*(true|false)', cleaned)
    ac = re.search(r'"action"\s*:\s*"([^"]*)"', cleaned)
    fc = re.search(r'"is_follow_command"\s*:\s*(true|false)', cleaned)
    cf = re.search(r'"confidence"\s*:\s*([0-9.]+)', cleaned)
    rs = re.search(r'"reasoning"\s*:\s*"([^"]*)"', cleaned)

    if nv:
        result["needs_vision"] = nv.group(1) == "true"
    if ac:
        result["action"] = ac.group(1)
    if fc:
        result["is_follow_command"] = fc.group(1) == "true"
    if cf:
        result["confidence"] = float(cf.group(1))
    if rs:
        result["reasoning"] = rs.group(1)

    # return if we got at least the critical fields
    if "needs_vision" in result and "is_follow_command" in result:
        return result

    return None


def normalise_action(action: str) -> str:
    """Normalise action aliases to valid action names."""
    if action is None:
        return None
    action = action.lower().strip()
    # apply alias mapping
    action = ACTION_ALIASES.get(action, action)
    # validate against known actions
    if action not in VALID_ACTIONS:
        print(f"[Router] Unknown action '{action}' — defaulting to stop")
        return "stop"
    return action


def parse_command(user_command: str) -> dict:
    """
    Send user command to local Ollama LLM and parse the JSON response.
    Returns a dict with needs_vision, action, is_follow_command, confidence, reasoning.
    """
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": SYSTEM_PROMPT + user_command,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 500,
            "num_ctx": 512,
            "stop": [
                "\n\nCommand:",
                "\nCommand:"
            ]
        }
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=150)
        response.raise_for_status()
        raw = response.json().get("response", "")
        print(f"[Router] Raw output: {raw!r}")

        parsed = extract_json(raw)

        if parsed:
            # enforce: if needs_vision=true, action must be null
            if parsed.get("needs_vision") or parsed.get("is_follow_command"):
                parsed["action"] = None

            # normalise action name
            parsed["action"] = normalise_action(parsed.get("action"))

            # ensure confidence is a float
            if parsed.get("confidence") is None:
                parsed["confidence"] = 0.9

            return parsed

        print("[Router] JSON extraction failed — falling back to keyword routing")
        return _keyword_fallback(user_command)

    except requests.exceptions.ConnectionError:
        print("[Router] Ollama not running — falling back to keyword routing")
        return _keyword_fallback(user_command)

    except requests.exceptions.Timeout:
        print("[Router] Ollama timeout — falling back to keyword routing")
        return _keyword_fallback(user_command)

    except Exception as e:
        print(f"[Router] Error: {e} — falling back to keyword routing")
        return _keyword_fallback(user_command)


def _keyword_fallback(user_command: str) -> dict:
    """
    Simple keyword-based fallback if Ollama is unavailable or fails.
    """
    cmd = user_command.lower()

    if any(kw in cmd for kw in ["follow", "track", "chase"]):
        return {"needs_vision": True,"action": None,"is_follow_command": True,"confidence": 0.9,"reasoning": "follow command requires visual tracking"}
    if any(kw in cmd for kw in ["find", "locate"]):
        return {"needs_vision": True,"action": None,"is_follow_command": False,"confidence": 0.9,"reasoning": "object must be located visually"}
    if any(kw in cmd for kw in ["sit", "sit down"]):
        return {"needs_vision": False, "action": "sit", "is_follow_command": False, "confidence": 0.9, "reasoning": "keyword: sit"}
    if any(kw in cmd for kw in ["stand up", "standup", "get up"]):
        return {"needs_vision": False, "action": "stand_up", "is_follow_command": False, "confidence": 0.9, "reasoning": "keyword: stand up"}
    if any(kw in cmd for kw in ["wave", "hello", "hi"]):
        return {"needs_vision": False, "action": "hello", "is_follow_command": False, "confidence": 0.9, "reasoning": "keyword: wave"}
    if any(kw in cmd for kw in ["dance"]):
        return {"needs_vision": False, "action": "dance1", "is_follow_command": False, "confidence": 0.9, "reasoning": "keyword: dance"}
    if any(kw in cmd for kw in ["stop", "halt", "freeze"]):
        return {"needs_vision": False, "action": "stop", "is_follow_command": False, "confidence": 0.9, "reasoning": "keyword: stop"}
    if any(kw in cmd for kw in ["faster", "speed up"]):
        return {"needs_vision": False, "action": "speed_fast", "is_follow_command": False, "confidence": 0.9, "reasoning": "keyword: faster"}
    if any(kw in cmd for kw in ["slower", "slow down"]):
        return {"needs_vision": False, "action": "speed_slow", "is_follow_command": False, "confidence": 0.9, "reasoning": "keyword: slower"}
    if any(kw in cmd for kw in ["backflip", "back flip"]):
        return {"needs_vision": False, "action": "back_flip", "is_follow_command": False, "confidence": 0.9, "reasoning": "keyword: backflip"}
    if any(kw in cmd for kw in ["forward", "move forward"]):
        return {"needs_vision": False, "action": "move_forward", "is_follow_command": False, "confidence": 0.9, "reasoning": "keyword: forward"}
    if any(kw in cmd for kw in ["back", "backward", "move back"]):
        return {"needs_vision": False, "action": "move_backward", "is_follow_command": False, "confidence": 0.9, "reasoning": "keyword: backward"}
    if any(kw in cmd for kw in ["stretch"]):
        return {"needs_vision": False, "action": "stretch", "is_follow_command": False, "confidence": 0.9, "reasoning": "keyword: stretch"}
    if any(kw in cmd for kw in ["pose"]):
        return {"needs_vision": False, "action": "pose", "is_follow_command": False, "confidence": 0.9, "reasoning": "keyword: pose"}
    if any(kw in cmd for kw in ["heart"]):
        return {"needs_vision": False, "action": "heart", "is_follow_command": False, "confidence": 0.9, "reasoning": "keyword: heart"}
    if any(kw in cmd for kw in ["trot"]):
        return {"needs_vision": False, "action": "trot_run", "is_follow_command": False, "confidence": 0.9, "reasoning": "keyword: trot"}

    # unknown — route to vision
    return {"needs_vision": True, "action": None, "is_follow_command": False, "confidence": 0.5, "reasoning": "unknown command — routing to vision"}


def _default_response() -> dict:
    return {
        "needs_vision": False,
        "action": "stop",
        "is_follow_command": False,
        "confidence": 0.0,
        "reasoning": "parse failure — defaulting to stop"
    }