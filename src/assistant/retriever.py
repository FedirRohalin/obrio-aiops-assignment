import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from src.assistant.kb import KnowledgeBase
from src.assistant.schemas import Article

# Регулярний вираз для видалення пунктуації (усе, що не є літерою, цифрою чи пробілом)
_PUNCTUATION_RE = re.compile(r"[^\w\s]", re.UNICODE)

# Величина бусту до балу, якщо токен запиту збігається з тегом статті
TAG_BOOST = 1.5


@dataclass
class ScoredArticle:
    """Стаття разом з фінальним балом релевантності."""

    article: Article
    score: float


class TicketRetriever:
    """Пошук релевантних статей бази знань за текстом тікета (BM25 + буст за тегами)."""

    def __init__(self, kb: KnowledgeBase) -> None:
        # Зберігаємо список статей з бази знань
        self._articles: list[Article] = kb.get_all_articles()

        # Попередньо обчислюємо множини токенів тегів для кожної статті
        self._tag_tokens: list[set[str]] = [
            self._extract_tag_tokens(article) for article in self._articles
        ]

        # Готуємо корпус: нормалізований текст + заголовок кожної статті
        corpus: list[list[str]] = [
            self._normalize(self._article_text(article)) for article in self._articles
        ]

        # BM25Okapi не можна ініціалізувати порожнім корпусом
        self._bm25: BM25Okapi | None = BM25Okapi(corpus) if corpus else None

    def _normalize(self, text: str) -> list[str]:
        # Нижній регістр, видалення пунктуації, розбиття на токени
        cleaned = _PUNCTUATION_RE.sub(" ", text.lower())
        return cleaned.split()

    def _article_text(self, article: Article) -> str:
        # Об'єднуємо заголовок та основний текст статті для індексування
        title = getattr(article, "title", "") or ""
        body = (
            getattr(article, "content", None)
            or getattr(article, "text", None)
            or getattr(article, "body", None)
            or ""
        )
        return f"{title} {body}".strip()

    def _extract_tag_tokens(self, article: Article) -> set[str]:
        # Повертаємо множину нормалізованих токенів з усіх тегів статті
        tokens: set[str] = set()
        for tag in getattr(article, "tags", None) or []:
            tokens.update(self._normalize(tag))
        return tokens

    def retrieve(
        self,
        text: str,
        min_score: float = 1.0,
        top_k: int = 2,
    ) -> list[ScoredArticle]:
        # Нормалізуємо запит; порожній запит або порожня база дають порожній результат
        query_tokens = self._normalize(text)
        if not query_tokens or self._bm25 is None or top_k <= 0:
            return []

        # Базові бали BM25 для всіх статей
        base_scores = self._bm25.get_scores(query_tokens)
        query_token_set = set(query_tokens)

        results: list[ScoredArticle] = []
        for article, tag_tokens, base_score in zip(
            self._articles, self._tag_tokens, base_scores
        ):
            score = float(base_score)

            # Буст, якщо хоча б один токен запиту збігається з тегом статті
            if query_token_set & tag_tokens:
                score += TAG_BOOST

            # Залишаємо лише статті з достатнім балом
            if score >= min_score:
                results.append(ScoredArticle(article=article, score=score))

        # Сортуємо за спаданням балу та повертаємо top_k найкращих
        results.sort(key=lambda item: item.score, reverse=True)
        return results[:top_k]
