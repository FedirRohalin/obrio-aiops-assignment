"""Base wrapper for interacting with LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from src.schemas import TicketClassificationResult


class LLMClient(ABC):
    """Abstract base class for model-provider integrations.

    Subclasses should implement provider-specific request/response handling,
    while this class defines the common interface used by the application.
    """

    def __init__(
        self,
        api_key: str | None,
        model_name: str,
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> None:
        """Initialize the common LLM client configuration.

        Args:
            api_key: API key used for authentication with the provider.
            model_name: Name of the model to call.
            temperature: Sampling temperature for generation.
            max_tokens: Maximum number of tokens to generate.
        """
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    @property
    def provider_name(self) -> str:
        """Return the provider class name."""
        return self.__class__.__name__

    def _prepare_messages(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
    ) -> list[dict[str, str]]:
        """Format a request as a list of message dictionaries.

        Args:
            prompt: Main user prompt or task description.
            system_prompt: Optional system-level instructions.

        Returns:
            A list of message objects ready to send to a provider.
        """
        messages: list[dict[str, str]] = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})
        return messages

    def _build_payload(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **extra_kwargs: Any,
    ) -> dict[str, Any]:
        """Build a provider-agnostic payload dictionary for the request.

        Args:
            messages: A list of chat messages.
            temperature: Optional override for generation temperature.
            max_tokens: Optional override for token limit.
            **extra_kwargs: Extra provider-specific request parameters.

        Returns:
            A dictionary suitable for serialization into the provider API body.
        """
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
        }

        if max_tokens is not None or self.max_tokens is not None:
            payload["max_tokens"] = max_tokens if max_tokens is not None else self.max_tokens

        payload.update(extra_kwargs)
        return payload

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Generate a text response from the configured model.

        Args:
            prompt: The task or instruction to send to the model.
            system_prompt: Optional system message to guide generation.
            temperature: Optional per-call temperature setting.
            max_tokens: Optional per-call response limit.

        Returns:
            The generated output as a single string.
        """

    @abstractmethod
    def classify_ticket(
        self,
        text: str,
        *,
        system_prompt: str | None = None,
    ) -> TicketClassificationResult:
        """Classify a support ticket into a typed result object.

        Args:
            text: Raw text content of the ticket or issue.
            system_prompt: Optional parser/instruction prompt.

        Returns:
            A validated classification result object.
        """


__all__ = ["LLMClient"]
