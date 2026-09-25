"""LLM client wrapper with strict JSON mode and error handling."""

import logging

from openai import (
    APIConnectionError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)

from src.config import MODEL_NAME

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT_SECONDS = 15.0


class LLMClientError(Exception):
    """Raised when the LLM call fails after all handling."""


class LLMClient:
    """Thin wrapper around the OpenAI chat completions API."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = MODEL_NAME,
    ) -> None:
        self._client = OpenAI(
            api_key=api_key,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        self._model_name = model_name

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Call the LLM and return the raw JSON string response."""
        try:
            response = self._client.chat.completions.create(
                model=self._model_name,
                temperature=0.0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
        except APITimeoutError as exc:
            logger.error("LLM request timed out: %s", exc)
            raise LLMClientError("LLM request timed out") from exc
        except RateLimitError as exc:
            logger.error("LLM rate limit exceeded: %s", exc)
            raise LLMClientError("LLM rate limit exceeded") from exc
        except APIConnectionError as exc:
            logger.error("LLM connection failed: %s", exc)
            raise LLMClientError("LLM connection failed") from exc

        content = response.choices[0].message.content
        if content is None:
            raise LLMClientError("LLM returned an empty response")
        return content
