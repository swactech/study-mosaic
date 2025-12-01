"""
Coverage agent: evaluates whether flashcard generation has sufficiently cited retrieved chunks.
"""

from typing import Dict, List

from app.config import settings
from app.context.prompts import COVERAGE_SYSTEM_PROMPT, CoverageResult

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


class CoverageAgent:
    """Agent that decides if coverage is sufficient."""

    def __init__(self):
        self.agent = LlmAgent(
            name="coverage_agent",
            instruction=COVERAGE_SYSTEM_PROMPT,
            model=settings.model_name,
            output_schema=CoverageResult,
            output_key="coverage_result",
        )

    def assess(
        self,
        total_chunk_ids: List[str],
        cited_chunk_ids: List[str],
        coverage_threshold: float,
    ) -> Dict:
        payload = {
            "total_chunks": total_chunk_ids,
            "cited_chunks": cited_chunk_ids,
            "coverage_threshold": coverage_threshold,
        }
        raw = self.agent.run(messages=[{"role": "user", "content": payload}])
        result = raw.output  # type: ignore[attr-defined]
        if isinstance(result, CoverageResult):
            return result.model_dump(by_alias=True)
        if isinstance(result, dict):
            return {
                "continue": result.get("continue") or result.get("continue_") or False,
                "coverage": result.get("coverage", len(cited_chunk_ids) / max(1, len(total_chunk_ids))),
                "missing": result.get("missing", [cid for cid in total_chunk_ids if cid not in cited_chunk_ids]),
            }
        return {
            "continue": False,
            "coverage": len(cited_chunk_ids) / max(1, len(total_chunk_ids)),
            "missing": [cid for cid in total_chunk_ids if cid not in cited_chunk_ids],
        }
