"""Configuration for the internal support assistant."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import dotenv

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ConfigError(ValueError):
    """Raised when an assistant setting is invalid or missing."""


def _env_str(name: str, default: str | None = None) -> str | None:
    """Return an environment variable, treating empty values as unset."""
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value


def _env_int(name: str, default: int | None = None) -> int | None:
    """Return and validate an integer environment variable."""
    raw_value = _env_str(name)
    if raw_value is None:
        return default

    try:
        return int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} has invalid value {raw_value!r}") from exc


def _env_float(name: str, default: float | None = None) -> float | None:
    """Return and validate a float environment variable."""
    raw_value = _env_str(name)
    if raw_value is None:
        return default

    try:
        return float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} has invalid value {raw_value!r}") from exc


def _resolve_path(value: str, default_value: str) -> Path:
    """Resolve a configured path relative to the project root."""
    candidate = Path(value or default_value)
    if not candidate.is_absolute():
        candidate = (PROJECT_ROOT / candidate).resolve()
    return candidate


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime settings for the internal support assistant."""

    openai_api_key: str | None = field(default=None, repr=False)
    model_name: str = "gpt-4o-mini"
    timeout_seconds: float = 15.0
    max_retries: int = 3
    retry_base_delay_seconds: float = 1.0
    total_deadline_seconds: float = 45.0
    max_output_tokens: int = 600
    max_ticket_chars: int = 1500
    retrieval_top_k: int = 2
    retrieval_min_score: float = 1.0
    kb_path: Path = field(
        default_factory=lambda: PROJECT_ROOT / "data" / "kb" / "articles.json"
    )
    eval_path: Path = field(
        default_factory=lambda: PROJECT_ROOT / "data" / "eval_tickets.json"
    )

    def __post_init__(self) -> None:
        """Validate all settings after initialization."""
        if self.timeout_seconds <= 0:
            raise ConfigError(
                f"ASSISTANT_TIMEOUT_SECONDS must be > 0, got {self.timeout_seconds!r}"
            )
        if not 0 <= self.max_retries <= 5:
            raise ConfigError(
                f"ASSISTANT_MAX_RETRIES must be between 0 and 5, got {self.max_retries!r}"
            )
        if self.retry_base_delay_seconds <= 0:
            raise ConfigError(
                "ASSISTANT_RETRY_BASE_DELAY must be > 0, "
                f"got {self.retry_base_delay_seconds!r}"
            )
        if self.total_deadline_seconds < self.timeout_seconds:
            raise ConfigError(
                "ASSISTANT_TOTAL_DEADLINE must be >= ASSISTANT_TIMEOUT_SECONDS, "
                f"got {self.total_deadline_seconds!r} < {self.timeout_seconds!r}"
            )
        if self.max_output_tokens <= 0:
            raise ConfigError(
                f"ASSISTANT_MAX_OUTPUT_TOKENS must be > 0, got {self.max_output_tokens!r}"
            )
        if self.max_ticket_chars <= 0:
            raise ConfigError(
                f"ASSISTANT_MAX_TICKET_CHARS must be > 0, got {self.max_ticket_chars!r}"
            )
        if self.retrieval_top_k < 1:
            raise ConfigError(
                f"ASSISTANT_TOP_K must be >= 1, got {self.retrieval_top_k!r}"
            )
        if self.retrieval_min_score < 0:
            raise ConfigError(
                "ASSISTANT_MIN_SCORE must be >= 0, " f"got {self.retrieval_min_score!r}"
            )

    def require_api_key(self) -> str:
        """Return the configured OpenAI API key or raise a configuration error."""
        if self.openai_api_key is None or self.openai_api_key == "":
            raise ConfigError("OPENAI_API_KEY is not set")
        return self.openai_api_key


def load_settings() -> Settings:
    """Load assistant settings from environment variables."""
    dotenv.load_dotenv(override=False)
    logger.debug("Loading assistant settings from the environment.")

    model_name = _env_str("ASSISTANT_MODEL", "gpt-4o-mini")
    timeout_seconds = _env_float("ASSISTANT_TIMEOUT_SECONDS", 15.0)
    max_retries = _env_int("ASSISTANT_MAX_RETRIES", 3)
    retry_base_delay_seconds = _env_float("ASSISTANT_RETRY_BASE_DELAY", 1.0)
    total_deadline_seconds = _env_float("ASSISTANT_TOTAL_DEADLINE", 45.0)
    max_output_tokens = _env_int("ASSISTANT_MAX_OUTPUT_TOKENS", 600)
    max_ticket_chars = _env_int("ASSISTANT_MAX_TICKET_CHARS", 1500)
    retrieval_top_k = _env_int("ASSISTANT_TOP_K", 2)
    retrieval_min_score = _env_float("ASSISTANT_MIN_SCORE", 1.0)
    kb_path = _resolve_path(
        _env_str("ASSISTANT_KB_PATH", "data/kb/articles.json"), "data/kb/articles.json"
    )
    eval_path = _resolve_path(
        _env_str("ASSISTANT_EVAL_PATH", "data/eval_tickets.json"),
        "data/eval_tickets.json",
    )

    if model_name is None or model_name == "":
        model_name = "gpt-4o-mini"

    if timeout_seconds is None:
        raise ConfigError("ASSISTANT_TIMEOUT_SECONDS has invalid value None")
    if max_retries is None:
        raise ConfigError("ASSISTANT_MAX_RETRIES has invalid value None")
    if retry_base_delay_seconds is None:
        raise ConfigError("ASSISTANT_RETRY_BASE_DELAY has invalid value None")
    if total_deadline_seconds is None:
        raise ConfigError("ASSISTANT_TOTAL_DEADLINE has invalid value None")
    if max_output_tokens is None:
        raise ConfigError("ASSISTANT_MAX_OUTPUT_TOKENS has invalid value None")
    if max_ticket_chars is None:
        raise ConfigError("ASSISTANT_MAX_TICKET_CHARS has invalid value None")
    if retrieval_top_k is None:
        raise ConfigError("ASSISTANT_TOP_K has invalid value None")
    if retrieval_min_score is None:
        raise ConfigError("ASSISTANT_MIN_SCORE has invalid value None")

    settings = Settings(
        openai_api_key=_env_str("OPENAI_API_KEY"),
        model_name=model_name,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        retry_base_delay_seconds=retry_base_delay_seconds,
        total_deadline_seconds=total_deadline_seconds,
        max_output_tokens=max_output_tokens,
        max_ticket_chars=max_ticket_chars,
        retrieval_top_k=retrieval_top_k,
        retrieval_min_score=retrieval_min_score,
        kb_path=kb_path,
        eval_path=eval_path,
    )

    return settings


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached copy of the loaded assistant settings."""
    return load_settings()


__all__ = [
    "ConfigError",
    "Settings",
    "get_settings",
    "load_settings",
]
