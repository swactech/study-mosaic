"""
Flashcard agent: pulls context via retrieval and drafts grounded flashcards.
"""

import json
from typing import List, Dict, Optional

from app.config import settings
from app.context.prompts import FlashcardSetStrict, FLASHCARD_SYSTEM_PROMPT

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
            output_schema=FlashcardSetStrict,
            output_key="flashcards",
        )

    def prepare_context(
        self,
        request: str,
        chunks: List[Dict],
    ) -> List[Dict]:
        """
        prepare context for flashcard agent
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
        return payload

    def parse_output(self, chunks, new_cards):
        total_chunk_ids = [c["id"] for c in chunks]
        covered: set = set()
        all_cards: List[Dict] = []
        existing_questions = {c.get("question") for c in all_cards}
        for card in new_cards:
            if card.get("question") in existing_questions:
                continue
            all_cards.append(card)
            existing_questions.add(card.get("question"))
            for cite in card.get("citations", []):
                cid = cite.get("location", {}).get("chunk_id")
                if cid:
                    covered.add(cid)

        coverage_ratio = len(covered) / len(total_chunk_ids)
        return {
            "flashcards": all_cards,
            "coverage": coverage_ratio,
            "cited_chunks": list(covered),
            "total_chunks": len(total_chunk_ids),
        }
