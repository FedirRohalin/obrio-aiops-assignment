"""Application configuration based on environment variables."""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY: str | None = os.getenv("ANTHROPIC_API_KEY")
MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-4o-mini")
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()


def get_api_key(provider_name: str) -> str | None:
    """Return the configured API key for a supported provider.

    Args:
        provider_name: Provider identifier such as "openai" or "anthropic".

    Returns:
        The API key value or ``None`` if it is not configured.

    Raises:
        ValueError: If the provider name is not supported.
    """
    normalized_name = provider_name.strip().lower()

    if normalized_name == "openai":
        return OPENAI_API_KEY
    if normalized_name == "anthropic":
        return ANTHROPIC_API_KEY

    raise ValueError(f"Unsupported provider: {provider_name!r}")


__all__ = [
    "ANTHROPIC_API_KEY",
    "LOG_LEVEL",
    "MODEL_NAME",
    "OPENAI_API_KEY",
    "get_api_key",
]
