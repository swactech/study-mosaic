"""
Coverage agent: evaluates whether flashcard generation has sufficiently cited retrieved chunks.
"""

import json
from typing import Any, Dict, List, Tuple

from app.config import settings
from app.context.prompts import COVERAGE_SYSTEM_PROMPT, CoverageResultStrict

# Prefer documented import path; fall back to legacy name if needed.
try:
    from google.adk.agents import LlmAgent  # type: ignore
    from google.adk.models import Gemini
    from google.adk.tools import FunctionTool
    from google.genai import types
except ImportError:  # pragma: no cover
    try:
        from adk import LlmAgent  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "google-adk is required (expected Agent in google.adk.agents). "
            "Install with `pip install google-adk`."
        ) from exc


def coerce_cards(raw_cards: Any) -> Tuple[List[Dict], List[str]]:
    errors: List[str] = []
    if not raw_cards:
        errors.append("No flashcards provided to coverage tool.")
        return [], errors
    if isinstance(raw_cards, dict) and "flashcards" in raw_cards:
        raw_cards = raw_cards.get("flashcards")
    if isinstance(raw_cards, str):
        try:
            parsed = json.loads(raw_cards)
            return (parsed if isinstance(parsed, list) else [parsed]), errors
        except json.JSONDecodeError:
            errors.append("Failed to parse flashcards JSON string.")
            return [], errors
    if isinstance(raw_cards, list):
        return raw_cards, errors
    if isinstance(raw_cards, dict):
        return [raw_cards], errors
    errors.append(
        f"Unsupported flashcard payload type: {type(raw_cards).__name__}")
    return [], errors


def extract_cited_ids(raw_cards: Any) -> Dict[str, Any]:
    """
    Extract chunk_id values from flashcard citations.

    Accepts multiple shapes:
    - list of flashcard dicts
    - single flashcard dict
    - {"flashcards": [...]} wrapper
    - JSON string representing any of the above

    Args:
        raw_cards: The list of flashcard dicts or single flashcard dict
                e.g., [{...}] or {"flashcards": [...]}.

    Returns:
        Dictionary with status and cited chunks.
        Success: {"status": "success", "cited_chunks": [...]}
        Error: {"status": "error", "error_message": "Failed to parse flashcard JSON string"}
    """
    cards, errors = coerce_cards(raw_cards)
    if errors and not cards:
        return {
            "status": "error",
            "error_message": "ERROR: " + " | ".join(errors),
        }

    cited = []
    for card in cards:
        for cite in card.get("citations", []):
            cid = cite.get("location", {}).get("chunk_id")
            if cid:
                cited.append(cid)
    return {"status": "success", "cited_chunks": cited}


class CoverageAgent:
    """Agent that decides if coverage is sufficient."""

    def __init__(self, coverage_threshold: float = 0.8):
        retry_config = types.HttpRetryOptions(
            attempts=5,  # Maximum retry attempts
            exp_base=7,  # Delay multiplier
            initial_delay=1,
            # Retry on these HTTP errors
            http_status_codes=[429, 500, 503, 504],
        )
        self.agent = LlmAgent(
            name="coverage_agent",
            description="Assess coverage from generation",
            instruction=COVERAGE_SYSTEM_PROMPT.format(
                coverage_threshold=coverage_threshold,
                coverage_schema=json.dumps(
                    CoverageResultStrict.model_json_schema(), indent=2)
            ),
            model=Gemini(
                model=settings.model_name,
                retry_options=retry_config
            ),
            tools=[FunctionTool(extract_cited_ids)],
            output_schema=CoverageResultStrict,
            output_key="coverage_result",
        )
