"""Ticket classification orchestration with safe fallback handling."""

import json
import logging

from pydantic import ValidationError

from src.llm_client import LLMClient, LLMClientError
from src.prompts import SYSTEM_PROMPT, build_user_prompt
from src.schemas import TicketCategory, TicketClassificationResult, TicketPriority

logger = logging.getLogger(__name__)

FALLBACK_REASON = "Fallback triggered: parsing/API failure"


class TicketClassifier:
    """Classifies support tickets using an LLM, with safe fallback on failure."""

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm_client = llm_client

    def classify(self, user_text: str) -> TicketClassificationResult:
        """Classify a ticket, falling back to manual review on any failure."""
        user_prompt = build_user_prompt(user_text)

        try:
            raw_response = self._llm_client.generate(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
            )
            return TicketClassificationResult.model_validate_json(raw_response)
        except LLMClientError as exc:
            logger.error("LLM call failed, using fallback: %s", exc)
        except json.JSONDecodeError as exc:
            logger.error("Invalid JSON from LLM, using fallback: %s", exc)
        except ValidationError as exc:
            logger.error("Schema validation failed, using fallback: %s", exc)

        return self._build_fallback_result()

    @staticmethod
    def _build_fallback_result() -> TicketClassificationResult:
        return TicketClassificationResult(
            category=TicketCategory.GENERAL_QUESTION,
            priority=TicketPriority.HIGH,
            recommended_action="Route to human agent for manual review",
            needs_human=True,
            reasoning=FALLBACK_REASON,
        )
