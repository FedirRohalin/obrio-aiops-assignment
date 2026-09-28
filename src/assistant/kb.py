import json
from pathlib import Path
from typing import Optional

from .schemas import Article


class KnowledgeBase:
    """Knowledge Base manager for Nebula support articles."""

    def __init__(self, file_path: str | Path) -> None:
        """
        Initialize the KnowledgeBase with a file path to the articles JSON.

        Args:
            file_path (str | Path): Path to the JSON file containing articles.
        """
        self.file_path = Path(file_path)
        self._articles: dict[str, Article] = {}

    def load(self) -> None:
        """Load and parse the JSON file into Article models."""
        with open(self.file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self._articles.clear()
        for item in data:
            article = Article.model_validate(item)
            self._articles[article.id] = article

    def get_article(self, article_id: str) -> Optional[Article]:
        """
        Retrieve an article by its ID.

        Args:
            article_id (str): The ID of the article to retrieve.

        Returns:
            Article | None: The matching Article object if found, else None.
        """
        return self._articles.get(article_id)

    def get_all_articles(self) -> list[Article]:
        """
        Return a list of all loaded articles.

        Returns:
            list[Article]: A list of all stored Article objects.
        """
        return list(self._articles.values())
