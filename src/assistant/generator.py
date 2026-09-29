import logging

from tenacity import RetryError

from src.assistant.llm_client import LLMClient
from src.assistant.prompts import SYSTEM_PROMPT, build_user_prompt
from src.assistant.resilience import (
    get_fallback_output,
    retry_on_logical_errors,
    retry_on_transient_errors,
)
from src.assistant.schemas import Article, AssistantOutput

logger = logging.getLogger(__name__)


class TicketGenerator:
    def __init__(self, client: LLMClient) -> None:
        self._client = client

    def generate_response(
        self, ticket_text: str, articles: list[Article]
    ) -> AssistantOutput:
        try:
            return self._generate_with_repair(ticket_text, articles)
        except RetryError:
            logger.error(
                "All retry attempts exhausted; returning fallback output.",
                exc_info=True,
            )
            return get_fallback_output()

    # Outer layer: exactly one repair retry for invalid JSON / schema / citation errors.
    # If the inner transient layer raises RetryError, it is not retried here and
    # propagates straight to generate_response.
    @retry_on_logical_errors
    def _generate_with_repair(
        self, ticket_text: str, articles: list[Article]
    ) -> AssistantOutput:
        user_prompt = build_user_prompt(ticket_text, articles)
        raw_data = self._call_llm(user_prompt)
        output = AssistantOutput(**raw_data)

        if output.citation is not None:
            article = next(
                (a for a in articles if a.id == output.citation.article_id), None
            )
            if article is None:
                raise ValueError("Hallucinated article_id")
            if output.citation.quote not in article.text:
                raise ValueError(
                    "Quote is not a verbatim substring of the article text"
                )

        return output

    # Inner layer: network/server errors only, exponential backoff + jitter, max 3 attempts.
    @retry_on_transient_errors
    def _call_llm(self, user_prompt: str) -> dict:
        return self._client.generate(SYSTEM_PROMPT, user_prompt)
