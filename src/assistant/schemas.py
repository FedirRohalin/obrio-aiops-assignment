from typing import Optional

from pydantic import BaseModel, ConfigDict


class Article(BaseModel):
    """Knowledge base article schema."""

    id: str
    title: str
    tags: list[str]
    text: str


class Citation(BaseModel):
    """Citation referencing a specific knowledge base article."""

    model_config = ConfigDict(extra="forbid")
    article_id: str
    quote: str


class AssistantOutput(BaseModel):
    """Output schema for the AI assistant."""

    model_config = ConfigDict(extra="forbid")
    summary: str
    formal_reply: str
    empathetic_reply: str
    short_reply: str
    citation: Optional[Citation]
