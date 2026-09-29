from __future__ import annotations

import json
import logging
from types import UnionType
from typing import Any, Literal, Union, get_args, get_origin

import openai
from pydantic import BaseModel, ValidationError
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from src.assistant.schemas import AssistantOutput

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Error classes
# ---------------------------------------------------------------------------

# Network / server-side failures: safe to retry with the same request.
TRANSIENT_ERRORS: tuple[type[BaseException], ...] = (
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.RateLimitError,
    openai.InternalServerError,
)

# Logical failures: the LLM answered, but the answer is unusable
# (invalid JSON, schema mismatch, hallucinated citation).
# NOTE: pydantic.ValidationError and json.JSONDecodeError are ValueError
# subclasses; they are listed explicitly for readability.
LOGICAL_ERRORS: tuple[type[BaseException], ...] = (
    ValidationError,
    json.JSONDecodeError,
    ValueError,
)

MAX_TRANSIENT_ATTEMPTS = 3
MAX_REPAIR_ATTEMPTS = 2  # 1 initial attempt + exactly 1 repair retry

# ---------------------------------------------------------------------------
# Retry decorators
# ---------------------------------------------------------------------------

# Exponential backoff with jitter: ~1s, ~2s, ... capped at 10s. Max 3 attempts.
# When attempts are exhausted, tenacity raises RetryError.
retry_on_transient_errors = retry(
    retry=retry_if_exception_type(TRANSIENT_ERRORS),
    stop=stop_after_attempt(MAX_TRANSIENT_ATTEMPTS),
    wait=wait_exponential_jitter(initial=1, max=10),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)

# Repair retry: exactly one extra attempt for logical errors, no delay needed
# (the failure is in the model output, not in the network).
retry_on_logical_errors = retry(
    retry=retry_if_exception_type(LOGICAL_ERRORS),
    stop=stop_after_attempt(MAX_REPAIR_ATTEMPTS),
    wait=wait_exponential_jitter(initial=0, max=0),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)

# ---------------------------------------------------------------------------
# Fallback output
# ---------------------------------------------------------------------------

FALLBACK_SUMMARY = "System unavailable. Please handle manually."
FALLBACK_REPLY = (
    "Thank you for contacting us. We are experiencing a temporary issue. "
    "Please wait a moment while a specialist reviews your request."
)


def _placeholder_for(annotation: Any) -> Any:
    """Build a neutral placeholder value for a field annotation."""
    origin = get_origin(annotation)

    if origin in (Union, UnionType):
        args = get_args(annotation)
        if type(None) in args:
            return None
        return _placeholder_for(args[0])

    if origin is Literal:
        return get_args(annotation)[0]

    if origin is list:
        (item_type,) = get_args(annotation) or (str,)
        return [_placeholder_for(item_type)]

    if annotation is str:
        return FALLBACK_REPLY

    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return _build_model(annotation)

    raise TypeError(f"Cannot build fallback placeholder for annotation: {annotation!r}")


def _build_model(
    model: type[BaseModel], overrides: dict[str, Any] | None = None
) -> Any:
    overrides = overrides or {}
    kwargs: dict[str, Any] = {}
    for name, field in model.model_fields.items():
        key = field.alias or name
        kwargs[key] = (
            overrides[name] if name in overrides else _placeholder_for(field.annotation)
        )
    return model(**kwargs)


def get_fallback_output() -> AssistantOutput:
    """Return a safe AssistantOutput used when the LLM pipeline is unavailable.

    summary is a fixed marker for the human agent, every reply field is a neutral
    "please wait" placeholder, and citation is None. Reply fields are filled by
    introspecting AssistantOutput, so this stays valid if the reply set changes.
    """
    return _build_model(
        AssistantOutput,
        overrides={"summary": FALLBACK_SUMMARY, "citation": None},
    )
