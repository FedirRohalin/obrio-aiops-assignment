from unittest.mock import MagicMock, patch

import pytest
from openai import APITimeoutError, RateLimitError

from src.assistant.generator import TicketGenerator
from src.assistant.resilience import get_fallback_output
from src.assistant.schemas import Article
from tests.assistant.fakes import FakeLLMClient

FALLBACK_SUMMARY = "System unavailable. Please handle manually."


@pytest.fixture
def articles() -> list[Article]:
    return [
        Article(id="KB-1", title="Test Article", tags=["test"], text="verbatim text")
    ]


def _valid_response(article_id: str = "KB-1", quote: str = "verbatim text") -> dict:
    """Build a schema-valid LLM response dict.

    Starts from the fallback payload so all reply fields match the real
    AssistantOutput schema, then overrides summary and citation.
    """
    data = get_fallback_output().model_dump(by_alias=True)
    data["summary"] = "Customer asks about the article."
    data["citation"] = {"article_id": article_id, "quote": quote}
    return data


@patch("time.sleep")
def test_transient_error_retry_success(
    mock_sleep: MagicMock, articles: list[Article]
) -> None:
    client = FakeLLMClient(
        [
            APITimeoutError(request=MagicMock()),
            _valid_response(),
        ]
    )
    generator = TicketGenerator(client)

    output = generator.generate_response("How do I do X?", articles)

    assert output.summary != FALLBACK_SUMMARY
    assert output.citation is not None
    assert output.citation.article_id == "KB-1"
    assert output.citation.quote == "verbatim text"
    assert client.responses == []
    mock_sleep.assert_called()


@patch("time.sleep")
def test_logical_error_repair_success(
    mock_sleep: MagicMock, articles: list[Article]
) -> None:
    client = FakeLLMClient(
        [
            _valid_response(article_id="BAD-ID"),
            _valid_response(article_id="KB-1"),
        ]
    )
    generator = TicketGenerator(client)

    output = generator.generate_response("How do I do X?", articles)

    assert output.summary != FALLBACK_SUMMARY
    assert output.citation is not None
    assert output.citation.article_id == "KB-1"
    assert client.responses == []


@patch("time.sleep")
def test_exhausted_retries_returns_fallback(
    mock_sleep: MagicMock, articles: list[Article]
) -> None:
    client = FakeLLMClient(
        [RateLimitError(message="x", response=MagicMock(), body=None) for _ in range(4)]
    )
    generator = TicketGenerator(client)

    output = generator.generate_response("How do I do X?", articles)

    assert output.summary == FALLBACK_SUMMARY
    assert output.citation is None
