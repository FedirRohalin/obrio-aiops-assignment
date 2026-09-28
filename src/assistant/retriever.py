import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from src.assistant.kb import KnowledgeBase
from src.assistant.schemas import Article

_PUNCTUATION_RE = re.compile(r"[^\w\s]", re.UNICODE)

TAG_BOOST = 1.5


@dataclass
class ScoredArticle:
    article: Article
    score: float


class TicketRetriever:
    def __init__(self, kb: KnowledgeBase) -> None:
        self._kb = kb
        self._articles: list[Article] = list(kb.articles)
        self._tag_tokens: list[set[str]] = [
            self._extract_tag_tokens(article) for article in self._articles
        ]
        corpus = [self._normalize(self._article_text(a)) for a in self._articles]
        self._bm25: BM25Okapi | None = BM25Okapi(corpus) if corpus else None


def _normalize(self, text: str) -> list[str]:
    cleaned = _PUNCTUATION_RE.sub(" ", text.lower())
    return cleaned.split()


def _article_text(self, article: Article) -> str:
    parts = [
        getattr(article, "title", "") or "",
        getattr(article, "content", "") or "",
    ]
    parts.extend(getattr(article, "tags", None) or [])
    return " ".join(parts)


def _extract_tag_tokens(self, article: Article) -> set[str]:
    tokens: set[str] = set()
    for tag in getattr(article, "tags", None) or []:
        tokens.update(self._normalize(tag))
    return tokens


def retrieve(
    self,
    text: str,
    top_k: int = 2,
    min_score: float = 1.0,
) -> list[ScoredArticle]:
    query_tokens = self._normalize(text)
    if not query_tokens or self._bm25 is None or top_k <= 0:
        return []

    base_scores = self._bm25.get_scores(query_tokens)
    query_token_set = set(query_tokens)

    scored: list[ScoredArticle] = []
    for article, tag_tokens, base_score in zip(
        self._articles, self._tag_tokens, base_scores
    ):
        score = float(base_score)
        if query_token_set & tag_tokens:
            score += TAG_BOOST
        if score >= min_score:
            scored.append(ScoredArticle(article=article, score=score))

    scored.sort(key=lambda item: item.score, reverse=True)
    return scored[:top_k]
