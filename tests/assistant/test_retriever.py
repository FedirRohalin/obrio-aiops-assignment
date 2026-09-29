"""Unit tests for TicketRetriever (BM25-based article retrieval).

All tests run fully in memory: the KnowledgeBase is mocked, so there is
no disk or network access, and BM25 scoring is deterministic.
"""

from unittest.mock import MagicMock

import pytest

from src.assistant.retriever import TicketRetriever
from src.assistant.schemas import Article

MIN_SCORE = 0.5


@pytest.fixture
def articles() -> list[Article]:
    return [
        Article(
            id="ID-1",
            title="Refund policy",
            tags=["refund", "money", "billing"],
            text=(
                "How to get a refund and receive your money back. "
                "Refunds are processed within 5 business days after "
                "the refund request is approved."
            ),
        ),
        Article(
            id="ID-2",
            title="App crash troubleshooting",
            tags=["bug", "crash", "freeze"],
            text=(
                "Technical troubleshooting when the app crashes or "
                "freezes on launch. Reinstall the app, clear the cache "
                "and update to the latest version to fix the crash."
            ),
        ),
        Article(
            id="ID-3",
            title="Cancel subscription",
            tags=["subscription", "cancel", "renewal"],
            text=(
                "How to cancel your subscription. Open settings, choose "
                "subscription and press cancel to stop future billing "
                "and renewal."
            ),
        ),
    ]


@pytest.fixture
def mock_kb(articles: list[Article]) -> MagicMock:
    """Mock KnowledgeBase exposing fake articles without touching disk/network."""
    kb = MagicMock(name="KnowledgeBase")
    kb.__iter__.side_effect = lambda: iter(articles)
    kb.__len__.return_value = len(articles)
    kb.articles = articles

    kb.get_all_articles.return_value = articles

    kb.load.return_value = None
    return kb


@pytest.fixture
def retriever(mock_kb: MagicMock) -> TicketRetriever:
    # Виправили: ініціалізуємо лише з mock_kb, без min_score
    return TicketRetriever(mock_kb)


@pytest.mark.parametrize(
    ("query", "expected_id"),
    [
        ("I want a refund, please give my money back", "ID-1"),
        ("The app crashes every time I launch it", "ID-2"),
    ],
    ids=["refund_query", "app_crash_query"],
)
def test_retrieve_hits(
    retriever: TicketRetriever, query: str, expected_id: str
) -> None:
    # Виправили: передаємо min_score безпосередньо під час пошуку
    results = retriever.retrieve(query, min_score=MIN_SCORE)

    assert results, f"Expected at least one result for query: {query!r}"
    assert results[0].article.id == expected_id


def test_retrieve_misses(retriever: TicketRetriever) -> None:
    # Виправили: передаємо min_score безпосередньо під час пошуку
    results = retriever.retrieve("asdfghjkl xyz", min_score=MIN_SCORE)

    assert results == []
