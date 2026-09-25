"""Pydantic schemas for ticket classification domain models."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class TicketCategory(str, Enum):
    """Supported categories for incoming support tickets."""

    BUG = "bug"
    FEATURE_REQUEST = "feature_request"
    QUESTION = "question"
    INCIDENT = "incident"
    DOCUMENTATION = "documentation"
    OTHER = "other"


class Priority(str, Enum):
    """Priority levels assigned to a classified ticket."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TicketClassificationResult(BaseModel):
    """Result object returned after classifying a support ticket.

    The model contains the resolved category and priority of a ticket, together
    with a confidence score and a human-readable summary.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )

    category: TicketCategory
    priority: Priority
    confidence: float = Field(..., ge=0.0, le=1.0)
    summary: str = Field(..., min_length=1, max_length=500)
    reasoning: str | None = Field(default=None, max_length=2000)


__all__ = ["Priority", "TicketCategory", "TicketClassificationResult"]
