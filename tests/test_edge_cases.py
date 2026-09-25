"""Integration tests for edge-case ticket classification behavior."""

import os

import pytest
from dotenv import load_dotenv

# Завантажуємо змінні оточення до локальних імпортів
load_dotenv()

from src.classifier import TicketClassifier  # noqa: E402
from src.llm_client import LLMClient  # noqa: E402
from src.schemas import TicketCategory, TicketPriority  # noqa: E402

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


@pytest.fixture
def classifier() -> TicketClassifier:
    llm_client = LLMClient(api_key=OPENAI_API_KEY)
    return TicketClassifier(llm_client)


@pytest.mark.skipif(not OPENAI_API_KEY, reason="OPENAI_API_KEY not set")
def test_aggression_needs_human(classifier: TicketClassifier) -> None:
    text = "This app is complete garbage, you scammers, delete my account now!"
    result = classifier.classify(text)
    assert result.needs_human is True


@pytest.mark.skipif(not OPENAI_API_KEY, reason="OPENAI_API_KEY not set")
def test_non_english_needs_human(classifier: TicketClassifier) -> None:
    text = "Quiero cancelar mi suscripción"
    result = classifier.classify(text)
    assert result.needs_human is True


@pytest.mark.skipif(not OPENAI_API_KEY, reason="OPENAI_API_KEY not set")
def test_safety_risk_critical_escalation(classifier: TicketClassifier) -> None:
    text = (
        "The tarot reading said I have no future. "
        "I'm so depressed I don't want to live anymore."
    )
    result = classifier.classify(text)
    assert result.category == TicketCategory.SAFETY_RISK
    assert result.priority == TicketPriority.CRITICAL
    assert result.needs_human is True
