"""
    Refiner agent: meets coverage requirement for generation
"""

from typing import List, Dict

from app.config import settings
from app.context.prompts import REFINER_SYSTEM_PROMPT

# Prefer documented import path; fall back to legacy name if needed.
try:
    from google.adk.agents import LlmAgent  # type: ignore
    from google.genai import types
    from google.adk.models import Gemini
    from google.adk.tools import FunctionTool
except ImportError:  # pragma: no cover
    try:
        from adk import LlmAgent  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "google-adk is required (expected Agent in google.adk.agents). "
            "Install with `pip install google-adk`."
        ) from exc


def exit_loop():
    """Call this function ONLY when the coverage is 'APPROVED', indicating the generation is finished and no more changes are needed."""
    return {"status": "approved", "message": "Coverage approved. Exiting refinement loop."}


class RefinerAgent:
    def __init__(self):
        retry_config = types.HttpRetryOptions(
            attempts=5,  # Maximum retry attempts
            exp_base=7,  # Delay multiplier
            initial_delay=1,
            # Retry on these HTTP errors
            http_status_codes=[429, 500, 503, 504],
        )

        self.agent = LlmAgent(
            name="refiner_agent",
            description="Refine generation to meet coverage",
            model=Gemini(
                model=settings.model_name,
                retry_options=retry_config
            ),
            instruction=REFINER_SYSTEM_PROMPT,
            tools=[FunctionTool(exit_loop)],
            output_key="refined_version"
        )
