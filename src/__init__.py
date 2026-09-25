"""Core package for the Nebula ticket classification service."""

from .config import ANTHROPIC_API_KEY, LOG_LEVEL, MODEL_NAME, OPENAI_API_KEY
from .llm_client import LLMClient
from .schemas import (
    TicketCategory,
    TicketClassificationResult,
    TicketInput,
    TicketPriority,
)

Priority = TicketPriority

__all__ = [
    "ANTHROPIC_API_KEY",
    "LOG_LEVEL",
    "LLMClient",
    "MODEL_NAME",
    "OPENAI_API_KEY",
    "Priority",
    "TicketCategory",
    "TicketClassificationResult",
    "TicketInput",
    "TicketPriority",
]
