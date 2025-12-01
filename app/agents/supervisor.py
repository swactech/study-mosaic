"""
Supervisor agent: routes user intents to specialized agents.
"""

from typing import Dict, List

from app.config import settings
from app.context.prompts import SUPERVISOR_SYSTEM_PROMPT
from app.agents.flashcards import FlashcardAgent
from app.agents.coverage import CoverageAgent
from app.tools.retrieval import retrieve_chunks

# Prefer documented import path; fall back to legacy name if needed.
try:
    from google.adk.agents import LlmAgent  # type: ignore
    from google.adk.agents import LoopAgent  # type: ignore
except ImportError:  # pragma: no cover
    try:
        from adk import LlmAgent  # type: ignore
        from adk import LoopAgent  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "google-adk is required (expected Agent in google.adk.agents). "
            "Install with `pip install google-adk`."
        ) from exc


class SupervisorAgent:
    """Lightweight router; currently only dispatches to flashcards."""

    def __init__(self):
        self.agent = LlmAgent(
            name="supervisor_agent",
            instruction=SUPERVISOR_SYSTEM_PROMPT,
            model=settings.model_name,
        )
        self.flashcard_agent = FlashcardAgent()
        self.coverage_agent = CoverageAgent()

    def handle(
        self,
        session_id: str,
        message: str,
        coverage_threshold: float = 0.8,
        max_rounds: int = 3,
        top_k: int = None,
    ) -> Dict:
        """
        Route to the correct sub-agent. For now, always go to flashcards.
        Note: as more generators (socratic, summaries, quizzes) are added,
        consider using an ADK Runner to manage multi-step workflows, logging,
        and state across agents/tools instead of manual loops.
        """
        # TODO: when more tasks are added, classify intent via supervisor model.
        chunks = retrieve_chunks(
            session_id=session_id, query=message, top_k=top_k)
        if not chunks:
            return {"flashcards": [], "coverage": 0.0, "cited_chunks": [], "total_chunks": 0}

        total_chunk_ids = [c["id"] for c in chunks]
        covered: set = set()
        all_cards: List[Dict] = []
        loop_agent = LoopAgent(
            name="FlashcardCoverageLoop",
            sub_agents=[self.flashcard_agent.agent, self.coverage_agent.agent],
            max_iterations=max_rounds,
        )

        iteration = 0
        # type: ignore[attr-defined]
        while iteration < loop_agent.max_iterations:
            iteration += 1
            new_cards = self.flashcard_agent.generate(
                session_id=session_id, request=message, chunks=chunks
            )
            # dedup by question text
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

            coverage_result = self.coverage_agent.assess(
                total_chunk_ids=total_chunk_ids,
                cited_chunk_ids=list(covered),
                coverage_threshold=coverage_threshold,
            )
            continue_flag = coverage_result.get("continue", False)
            if not continue_flag:
                break

        coverage_ratio = len(covered) / len(total_chunk_ids)
        return {
            "flashcards": all_cards,
            "coverage": coverage_ratio,
            "cited_chunks": list(covered),
            "total_chunks": len(total_chunk_ids),
        }
