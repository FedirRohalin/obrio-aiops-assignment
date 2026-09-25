"""Pydantic schema models used for ticket validation and LLM output."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class TicketCategory(str, Enum):
    """Supported categories used in the support ticket dataset."""

    SUBSCRIPTION = "Subscription"
    REFUND = "Refund"
    BUG = "Bug"
    EXPERT_COMPLAINT = "Expert_Complaint"
    GENERAL_QUESTION = "General_Question"
    SAFETY_RISK = "Safety_Risk"


class TicketPriority(str, Enum):
    """Priority levels used for issue triage and escalation."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class TicketClassificationResult(BaseModel):
    """Structured response returned by the model for a ticket classification.

    The schema is intentionally compact and suitable for downstream support
    workflow automation, including routing and escalation decisions.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )

    category: TicketCategory
    priority: TicketPriority
    recommended_action: str = Field(
        ...,
        min_length=1,
        max_length=80,
        description="Short action for support staff, max 10 words.",
    )
    needs_human: bool = Field(
        default=False,
        description="Whether the ticket should be escalated to a human agent.",
    )
    reasoning: str = Field(
        ...,
        min_length=1,
        max_length=400,
        description="Short explanation of the model decision in 1-2 sentences.",
    )


class TicketInput(BaseModel):
    """Input schema for validating tickets from the dataset file."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )

    ticket_id: str = Field(..., min_length=1, max_length=200)
    user_text: str = Field(..., min_length=1)
    expected_category: Optional[TicketCategory] = None
    expected_priority: Optional[TicketPriority] = None
    is_edge_case: bool = False
    notes: Optional[str] = Field(default=None, max_length=1000)


__all__ = [
    "TicketCategory",
    "TicketClassificationResult",
    "TicketInput",
    "TicketPriority",
]
