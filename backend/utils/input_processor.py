# backend/utils/input_processor.py

MAX_LENGTH = 200

VALID_ACTIONS = [
    "follow", "find", "search", "go to", "move",
    "stop", "sit", "stand", "wave", "emote",
    "describe", "look", "watch", "track",
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
