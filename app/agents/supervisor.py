"""
Supervisor agent: routes user intents to specialized agents.
"""

import json
from typing import Any, AsyncGenerator, Dict, List

from app.agents.flashcards import FlashcardAgent
from app.agents.coverage import CoverageAgent
from app.tools.retrieval import retrieve_chunks
from app.adk_utils import run_agent_and_get_state

# Prefer documented import path; fall back to legacy name if needed.
try:
    from google.adk.agents import BaseAgent, LoopAgent  # type: ignore
    from google.adk.agents.invocation_context import InvocationContext  # type: ignore
except ImportError:  # pragma: no cover
    try:
        from adk import BaseAgent, LoopAgent  # type: ignore
        from adk import InvocationContext  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "google-adk is required (expected Agent in google.adk.agents). "
            "Install with `pip install google-adk`."
        ) from exc


class SupervisorAgent(BaseAgent):
    """Custom orchestrator that loops flashcard + coverage agents via a runner."""

    # Declare fields so Pydantic/BaseAgent allows setting them.
    max_iterations: int = 3
    coverage_threshold: float = 0.8
    flashcard: FlashcardAgent
    coverage: CoverageAgent
    flashcard_agent: Any
    coverage_agent: Any
    loop_agent: Any

    def __init__(self, max_iterations: int = 3, coverage_threshold: float = 0.8):
        # Child agents that do the real work.
        flashcard = FlashcardAgent()
        coverage = CoverageAgent(coverage_threshold)
        flashcard_agent = flashcard.agent
        coverage_agent = coverage.agent

        # LoopAgent is registered for introspection; control flow is managed here.
        loop_agent = LoopAgent(
            name="flashcard_coverage_loop",
            sub_agents=[flashcard_agent, coverage_agent],
            max_iterations=max_iterations,
        )

        super().__init__(
            name="supervisor_orchestrator",
            # Only attach the loop agent to avoid double-parenting children.
            sub_agents=[loop_agent],
            max_iterations=max_iterations,
            coverage_threshold=coverage_threshold,
            flashcard=flashcard,
            coverage=coverage,
            flashcard_agent=flashcard_agent,
            coverage_agent=coverage_agent,
            loop_agent=loop_agent,
        )

    @staticmethod
    def _coerce_cards(raw_cards: Any) -> List[Dict]:
        if not raw_cards:
            return []
        if isinstance(raw_cards, str):
            try:
                parsed = json.loads(raw_cards)
                return parsed if isinstance(parsed, list) else [parsed]
            except json.JSONDecodeError:
                return []
        if isinstance(raw_cards, list):
            return raw_cards
        if isinstance(raw_cards, dict):
            return [raw_cards]
        return []

    @staticmethod
    def _extract_cited_ids(cards: List[Dict]) -> List[str]:
        cited = []
        for card in cards:
            for cite in card.get("citations", []):
                cid = cite.get("location", {}).get("chunk_id")
                if cid:
                    cited.append(cid)
        return cited

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Any, None]:
        """
        Manual loop: run flashcards, parse citations, run coverage, repeat if needed.
        """
        iteration = 0
        total_chunks = ctx.session.state.get("total_chunks", [])
        coverage_threshold = ctx.session.state.get(
            "coverage_threshold", self.coverage_threshold
        )

        while iteration < self.loop_agent.max_iterations:
            iteration += 1

            # Run flashcards; their output is stored in session via output_key.
            async for event in self.flashcard_agent.run_async(ctx):
                yield event

            cards = self._coerce_cards(ctx.session.state.get("flashcards"))
            cited_ids = self._extract_cited_ids(cards)

            coverage_payload = {
                "total_chunks": total_chunks,
                "cited_chunks": cited_ids,
                "coverage_threshold": coverage_threshold,
            }
            # Run coverage synchronously; store result in session state.
            raw = self.coverage_agent.run(
                messages=[
                    {"role": "user", "content": json.dumps(coverage_payload)}]
            )
            coverage_output = getattr(raw, "output", raw)
            ctx.session.state["coverage_result"] = coverage_output

            continue_flag = False
            if isinstance(coverage_output, dict):
                continue_flag = coverage_output.get("continue") or coverage_output.get(
                    "continue_", False
                )

            if not continue_flag:
                break

    def handle(
        self,
        session_id: str,
        message: str,
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

        flashcard_context = self.flashcard.prepare_context(
            request=message, chunks=chunks
        )
        # Run the orchestrator (this agent) via InMemoryRunner so loop control stays here.
        total_chunk_ids = [c["id"] for c in chunks]
        session_state = run_agent_and_get_state(
            agent=self,
            payload_text=json.dumps(flashcard_context),
            session_id=session_id,
            initial_state={
                "total_chunks": total_chunk_ids,
                "coverage_threshold": self.coverage_threshold,
            },
        )

        raw_cards = session_state.get("flashcards") or []
        if isinstance(raw_cards, str):
            try:
                raw_cards = json.loads(raw_cards)
            except json.JSONDecodeError:
                raw_cards = {}

        result = self.flashcard.parse_output(
            chunks, raw_cards.get("flashcards", []))

        coverage_state = session_state.get("coverage_result")
        if isinstance(coverage_state, str):
            try:
                coverage_state = json.loads(coverage_state)
            except json.JSONDecodeError:
                coverage_state = None
        if isinstance(coverage_state, dict):
            result["coverage"] = coverage_state.get(
                "coverage", result["coverage"])
        return result
