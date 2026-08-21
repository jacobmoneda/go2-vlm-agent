class SharedState:
    def __init__(self):
        self.base_prompt = (
            "You are a robot vision system. Given the image and task, respond with ONLY a JSON object. "
            "No explanation, no markdown, no extra text. "
            'Format: {"action": "<move_forward|move_left|move_right|stop|search|wait|emote_wave|emote_sit|emote_dance>", '
            '"target_visible": <true|false>, '
            '"target_description": "<string or null>", '
            '"reasoning": "<one sentence>"}'
        )
        self.user_prompt = ""
        self.latest_result = None
        self.camera = None

    @property
    def latest_prompt(self):
        if self.user_prompt:
            return f"{self.base_prompt} Instruction: {self.user_prompt}"
        return self.base_prompt

shared_state = SharedState()


