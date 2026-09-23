# backend/utils/input_processor.py

MAX_LENGTH = 200

VALID_ACTIONS = [
    "follow", "find", "search", "go to", "move",
    "stop", "sit", "stand", "wave", "emote",
    "describe", "look", "watch", "track",
    "detect", "see",
]

BLOCKED_PHRASES = [
    "ignore previous",
    "ignore instructions",
    "system prompt",
    "jailbreak",
    "disregard",
    "forget instructions",
]


class InvalidPromptError(Exception):
    pass


def preprocess_prompt(raw: str) -> str:
    """
    Cleans and validates a user prompt before passing to the VLM.
    Returns the cleaned prompt or raises InvalidPromptError.
    """

    # 1. Strip whitespace
    prompt = raw.strip()

    # 2. Reject empty
    if not prompt:
        raise InvalidPromptError("Prompt cannot be empty.")

    # 3. Reject too long
    if len(prompt) > MAX_LENGTH:
        raise InvalidPromptError(f"Prompt too long. Max {MAX_LENGTH} characters, got {len(prompt)}.")

    # 4. Block prompt injection attempts
    lower = prompt.lower()
    for phrase in BLOCKED_PHRASES:
        if phrase in lower:
            raise InvalidPromptError("Prompt contains disallowed content.")

    # 5. Check prompt contains a recognisable action
    if not any(action in lower for action in VALID_ACTIONS):
        raise InvalidPromptError(
            f"Prompt must contain a valid action. Valid actions: {', '.join(VALID_ACTIONS)}."
        )

    # 6. Normalise — strip extra whitespace between words
    prompt = " ".join(prompt.split())

    return prompt

def process_input(raw: str) -> dict:
    """
    Cleans, validates, and parses a user prompt into a structured task dict.
    Returns {"action": str, "target": str or None}
    """
    # first clean and validate
    prompt = preprocess_prompt(raw)
    lower = prompt.lower()

    # extract action
    action = None
    for valid_action in VALID_ACTIONS:
        if valid_action in lower:
            action = valid_action
            break

    # extract target — word(s) after the action
    target = None
    if action:
        # find everything after the action keyword
        after_action = lower.split(action, 1)[-1].strip()

        # strip common filler words
        filler = ["the", "a", "an", "me", "my", "this", "that"]
        words = [w for w in after_action.split() if w not in filler]

        if words:
            target = words[0]  # take first meaningful word as target

    return {
        "action": action,
        "target": target,
        "raw": prompt
    }