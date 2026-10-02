# backend/utils/command_router.py
import json
import re
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "phi3-fast"
SYSTEM_PROMPT = """You route natural-language commands for a Unitree Go2 robot.

Return ONLY one JSON object:
{"needs_vision": false, "action": null, "is_follow_command": false, "confidence": 0.0}

DIRECT:
If the command can be performed without the camera:
needs_vision=false, is_follow_command=false, action=<robot action>.

Valid actions:
move_forward, move_backward, move_left, move_right,
turn_left, turn_right, stop, sit, stand_up, stand_down,
hello, dance1, dance2, stretch, pose, heart,
front_flip, back_flip, trot_run,
speed_slow, speed_normal, speed_fast

Interpret natural language by meaning. Examples:
wave/greet/say hello -> hello
dance -> dance1
go forward/come forward -> move_forward
go back -> move_backward
step left/right -> move_left/move_right
turn left/right -> turn_left/turn_right
speed up/faster -> speed_fast
slow down/slower -> speed_slow
normal speed -> speed_normal
halt/freeze -> stop
get up -> stand_up

FOLLOW:
If asked to continuously follow, track, or chase a target:
needs_vision=true, is_follow_command=true, action=null.

VISION:
If the answer requires seeing the camera, such as identifying,
describing, finding, locating, or checking whether something is visible:
needs_vision=true, is_follow_command=false, action=null.

Important:
"move left" = move_left
"turn left" = turn_left
"find the bottle" = vision
"follow the person" = follow
"stop following" = stop

If uncertain, use vision. Never invent an action.

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
    "sit_down": "sit",
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
    Route command: keyword fallback first, LLM only for complex/unknown commands.
    """
    cmd = user_command.lower()

    # check if any keyword matches — instant, no LLM needed
    keyword_result = _keyword_fallback(user_command)

    # keyword fallback returns confidence 0.5 for unknown commands
    # anything above 0.5 means a keyword matched — return immediately
    if keyword_result.get("confidence", 0) > 0.5:
        print(f"[Router] Keyword match: {keyword_result.get('action')} | {keyword_result.get('reasoning')}")
        return keyword_result

    # unknown command — run through LLM
    print(f"[Router] No keyword match for '{user_command}' — routing to LLM")
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": SYSTEM_PROMPT + user_command,
        "stream": False,
        "keep_alive": -1,
        "options": {
            "temperature": 0.0,
            "num_predict": 40,
            "num_ctx": 256,
            "stop": ["\n\nCommand:", "\nCommand:", "}\n"]
        }
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=10)
        response.raise_for_status()
        raw = response.json().get("response", "")
        print(f"[Router] LLM output: {raw!r}")

        parsed = extract_json(raw)

        if parsed:
            if parsed.get("needs_vision") or parsed.get("is_follow_command"):
                parsed["action"] = None
            parsed["action"] = normalise_action(parsed.get("action"))
            if parsed.get("confidence") is None:
                parsed["confidence"] = 0.9
            return parsed

        print("[Router] LLM JSON extraction failed — defaulting to vision")
        return {"needs_vision": True, "action": None, "is_follow_command": False, "confidence": 0.5, "reasoning": "LLM parse failed"}

    except requests.exceptions.ConnectionError:
        print("[Router] Ollama not running — defaulting to vision")
        return {"needs_vision": True, "action": None, "is_follow_command": False, "confidence": 0.5, "reasoning": "Ollama unavailable"}

    except requests.exceptions.Timeout:
        print("[Router] LLM timeout — defaulting to vision")
        return {"needs_vision": True, "action": None, "is_follow_command": False, "confidence": 0.5, "reasoning": "LLM timeout"}

    except Exception as e:
        print(f"[Router] Error: {e} — defaulting to vision")
        return {"needs_vision": True, "action": None, "is_follow_command": False, "confidence": 0.5, "reasoning": "error"}


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