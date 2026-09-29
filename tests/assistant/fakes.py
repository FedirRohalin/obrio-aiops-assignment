from typing import Any


class FakeLLMClient:
    """Test double for LLMClient that replays a scripted sequence of outcomes.

    Each item in `responses` is either a dict (returned as the LLM output)
    or an exception instance (raised when that call happens).
    """

    def __init__(self, responses: list[Any]) -> None:
        self.responses = responses

    def generate(self, system_prompt: str, user_prompt: str) -> dict:
        item = self.responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item
