"""
Supervisor agent: routes user intents to specialized agents.
"""

import json
from typing import Any, AsyncGenerator, Dict

from app.agents.flashcards import FlashcardAgent
from app.agents.coverage import CoverageAgent
from app.agents.refiner import RefinerAgent
from app.tools.retrieval import retrieve_chunks
from app.adk_utils import run_agent_and_get_state

# Prefer documented import path; fall back to legacy name if needed.
try:
    from google.adk.agents import BaseAgent, LoopAgent, SequentialAgent  # type: ignore
    from google.adk.agents.invocation_context import InvocationContext
except ImportError:  # pragma: no cover
    try:
        from adk import BaseAgent, LoopAgent, SequentialAgent  # type: ignore
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
    sequence_agent: Any

    def __init__(self, max_iterations: int = 3, coverage_threshold: float = 0.8):
        # Child agents that do the real work.
        flashcard = FlashcardAgent()
        coverage = CoverageAgent(coverage_threshold)
        refiner = RefinerAgent()
        flashcard_agent = flashcard.agent
        coverage_agent = coverage.agent
        refiner_agent = refiner.agent

        loop_agent = LoopAgent(
            name="flashcard_coverage_loop",
            sub_agents=[coverage_agent, refiner_agent],
            max_iterations=max_iterations,
        )

        sequence_agent = SequentialAgent(
            name="flashcard_generation_pipeline",
            sub_agents=[flashcard_agent, loop_agent]
        )

        super().__init__(
            name="supervisor_orchestrator",
            sub_agents=[sequence_agent],
            max_iterations=max_iterations,
            coverage_threshold=coverage_threshold,
            flashcard=flashcard,
            coverage=coverage,
            flashcard_agent=flashcard_agent,
            coverage_agent=coverage_agent,
            loop_agent=loop_agent,
            sequence_agent=sequence_agent,
        )

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Any, None]:
        # Delegate orchestration to the pre-built SequentialAgent.
        sequence_agent = getattr(self, "sequence_agent", None) or (
            self.sub_agents[0] if self.sub_agents else None
        )
        if sequence_agent is None:
            return

        async for event in sequence_agent.run_async(ctx):
            yield event

        print(f"\n\n{ctx.session.state.keys()}\n\n")

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

        raw_cards = session_state.get(
            "refined_version", session_state.get("flashcards", {}).get("flashcards")) or []
        if isinstance(raw_cards, str):
            try:
                raw_cards = json.loads(raw_cards)
            except json.JSONDecodeError:
                raw_cards = {}

        result = self.flashcard.parse_output(
            chunks, raw_cards)

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
