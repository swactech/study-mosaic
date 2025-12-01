"""
Flashcard agent: pulls context via retrieval and drafts grounded flashcards.
"""

import json
from typing import List, Dict, Optional

from app.config import settings
from app.context.prompts import FlashcardSet, FLASHCARD_SYSTEM_PROMPT

# Prefer documented import path; fall back to legacy name if needed.
try:
    from google.adk.agents import LlmAgent  # type: ignore
except ImportError:  # pragma: no cover
    try:
        from adk import LlmAgent  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "google-adk is required (expected Agent in google.adk.agents). "
            "Install with `pip install google-adk`."
        ) from exc


class FlashcardAgent:
    """Specialized agent for flashcard generation."""

    def __init__(self):
        self.agent = LlmAgent(
            name="flashcard_agent",
            description="Generates grounded flashcards with citations.",
            instruction=FLASHCARD_SYSTEM_PROMPT,
            model=settings.model_name,
            output_schema=FlashcardSet,
            output_key="flashcards",
        )

    def generate(
        self,
        session_id: str,
        request: str,
        chunks: List[Dict],
    ) -> List[Dict]:
        """
        Generate flashcards for the given request using provided chunks as context.
        """
        if not chunks:
            return []

        context_parts = []
        for chunk in chunks:
            context_parts.append(
                f"[chunk_id={chunk['id']} page={chunk.get('page')}] {chunk['text']}"
            )
        context_blob = "\n\n".join(context_parts)

        payload = {
            "task": "flashcards",
            "request": request,
            "context": context_blob,
        }
        raw_response = self.agent.run(
            messages=[{"role": "user", "content": json.dumps(payload)}]
        )
        cards = raw_response.output  # type: ignore[attr-defined]
        return cards if isinstance(cards, list) else [cards]
