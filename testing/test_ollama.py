# test_ollama.py
import requests
import json
import re

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "phi3"

test_commands = [
    "follow the person",
    "sit down",
    "wave hello",
    "do a backflip",
    "stop",
    "go faster",
    "what do you see",
    "stop when you see a chair",
    "dance",
    "move forward",
]

# keep prompt as short as possible to reduce context bleeding
SYSTEM_PROMPT = """You are a robot command parser. Output ONLY a JSON object, no markdown, no explanation.

Valid actions: move_forward, move_backward, move_left, move_right, turn_left, turn_right, stop, sit, stand_up, stand_down, hello, dance1, dance2, stretch, pose, heart, front_flip, back_flip, trot_run, speed_slow, speed_normal, speed_fast, search

JSON format: {"needs_vision": bool, "action": "action or null", "is_follow_command": bool, "confidence": float, "reasoning": "one sentence"}

Rules:
- needs_vision=true only if the command requires seeing the camera to decide
- is_follow_command=true for follow/track/chase/find commands
- if is_follow_command=true, set action=null
- if needs_vision=true, set action=null

Command: """


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


def call_ollama(command: str) -> dict:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": SYSTEM_PROMPT + command,
        "stream": False,
        "options": {
            "temperature": 0.0,    # fully deterministic
            "num_predict": 300,    # enough for full JSON
            "num_ctx": 512         # small context window — prevents bleeding
        }
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=30)
        response.raise_for_status()
        raw = response.json().get("response", "").strip()
        parsed = extract_json(raw)

        if parsed:
            return {"raw": raw, "parsed": parsed, "error": None}
        else:
            return {"raw": raw, "parsed": None, "error": "JSON extract failed"}

    except requests.exceptions.ConnectionError:
        return {"raw": None, "parsed": None, "error": "Ollama not running — start with: ollama serve"}
    except requests.exceptions.Timeout:
        return {"raw": None, "parsed": None, "error": "Request timed out"}
    except Exception as e:
        return {"raw": None, "parsed": None, "error": str(e)}


if __name__ == "__main__":
    print(f"Testing Ollama with model: {OLLAMA_MODEL}")
    print(f"URL: {OLLAMA_URL}")
    print("=" * 60)

    pass_count = 0
    fail_count = 0

    for cmd in test_commands:
        print(f"\nCommand: '{cmd}'")
        result = call_ollama(cmd)

        if result["error"] and result["parsed"] is None:
            print(f"  ERROR: {result['error']}")
            if result["raw"]:
                print(f"  RAW: {result['raw'][:200]}")
            fail_count += 1
            continue

        parsed = result["parsed"]
        if parsed:
            print(f"  needs_vision:      {parsed.get('needs_vision')}")
            print(f"  action:            {parsed.get('action')}")
            print(f"  is_follow_command: {parsed.get('is_follow_command')}")
            print(f"  confidence:        {parsed.get('confidence')}")
            print(f"  reasoning:         {parsed.get('reasoning')}")
            pass_count += 1
        else:
            print(f"  RAW OUTPUT: {result['raw']}")
            fail_count += 1

    print("\n" + "=" * 60)
    print(f"Results: {pass_count} passed | {fail_count} failed")