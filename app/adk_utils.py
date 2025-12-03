"""
Utility to import the Google ADK package regardless of module name.

The pip package is `google-adk`. Some environments expose it as `google.adk`,
others as `adk`. This helper tries both to make the scaffold resilient.
"""

import importlib
import json
import asyncio
import inspect
from typing import Any, Optional


def load_adk():
    """Return the imported ADK module or raise a helpful error."""
    for module_name in ("google.adk", "adk"):
        try:
            return importlib.import_module(module_name)
        except ImportError:
            continue
    raise ImportError(
        "google-adk is required but not installed or importable. "
        "Install with `pip install google-adk`."
    )


def run_with_inmemory_runner(
    agent: Any,
    payload_text: str,
    session_id: str,
    *,
    app_name: str = "study-mosaic",
    user_id: Optional[str] = None,
) -> Any:
    """
    Run an ADK agent via InMemoryRunner and return the stored output.

    ADK v0.4+ requires a Runner to execute agents; there is no direct `agent.run`.
    This helper spins up an InMemoryRunner, ensures the session exists, sends a
    single user message, and then pulls the agent's `output_key` from session state.
    """
    load_adk()  # Ensure helpful error if ADK is missing.

    try:
        from google.adk.runners import InMemoryRunner  # type: ignore
    except ImportError:  # pragma: no cover - fallback
        try:
            from adk import InMemoryRunner  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "google-adk is required (expected InMemoryRunner). "
                "Install with `pip install google-adk`."
            ) from exc

    try:
        from google.genai import types as genai_types  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "google-generativeai is required for ADK message types. "
            "Install with `pip install google-generativeai`."
        ) from exc

    def _maybe_await(obj):
        if inspect.iscoroutine(obj):
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                return asyncio.run(obj)
            return loop.run_until_complete(obj)
        return obj

    user = user_id or session_id
    runner = InMemoryRunner(agent=agent, app_name=app_name)
    session_service = runner.session_service

    # Ensure the session exists before running.
    session = _maybe_await(
        session_service.get_session(
            app_name=app_name, user_id=user, session_id=session_id)
    )
    if not session:
        session = _maybe_await(
            session_service.create_session(
                app_name=app_name, user_id=user, session_id=session_id)
        )

    # Build ADK content message.
    content = genai_types.Content(
        role="user", parts=[genai_types.Part(text=payload_text)])

    # Drive the runner to completion (synchronous generator).
    for _ in runner.run(user_id=user, session_id=session_id, new_message=content):
        pass

    session = _maybe_await(
        session_service.get_session(
            app_name=app_name, user_id=user, session_id=session_id)
    )
    if not session:
        return None

    output_key = getattr(agent, "output_key", None)
    raw_output = session.state.get(output_key) if output_key else None

    if isinstance(raw_output, str):
        try:
            return json.loads(raw_output)
        except json.JSONDecodeError:
            return raw_output
    return raw_output


def run_agent_and_get_state(
    agent: Any,
    payload_text: str,
    session_id: str,
    *,
    app_name: str = "study-mosaic",
    user_id: Optional[str] = None,
    initial_state: Optional[dict] = None,
):
    """
    Run an ADK agent via InMemoryRunner and return the full session state.
    Useful when multiple sub-agents emit different output_keys (e.g., flashcards + coverage).
    """
    load_adk()  # validate ADK import

    try:
        from google.adk.runners import InMemoryRunner  # type: ignore
    except ImportError:  # pragma: no cover
        try:
            from adk import InMemoryRunner  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "google-adk is required (expected InMemoryRunner). "
                "Install with `pip install google-adk`."
            ) from exc

    try:
        from google.genai import types as genai_types  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "google-generativeai is required for ADK message types. "
            "Install with `pip install google-generativeai`."
        ) from exc

    def _maybe_await(obj):
        if inspect.iscoroutine(obj):
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                return asyncio.run(obj)
            return loop.run_until_complete(obj)
        return obj

    user = user_id or session_id
    runner = InMemoryRunner(agent=agent, app_name=app_name)
    session_service = runner.session_service

    session = _maybe_await(
        session_service.get_session(
            app_name=app_name, user_id=user, session_id=session_id)
    )
    if not session:
        session = _maybe_await(
            session_service.create_session(
                app_name=app_name, user_id=user, session_id=session_id, state=initial_state
            )
        )
    elif initial_state:
        # Persist initial state for existing sessions so LLM templates can reference it.
        session.state.update(initial_state)
        # Best-effort: update the backing session store if the service exposes it (InMemoryRunner does).
        try:
            storage_session = session_service.sessions[app_name][user][session_id]
            storage_session.state.update(initial_state)
        except Exception:
            # If the session service doesn't expose in-memory storage, we still have the local update.
            pass

    content = genai_types.Content(
        role="user", parts=[genai_types.Part(text=payload_text)])
    for _ in runner.run(user_id=user, session_id=session_id, new_message=content):
        # print(f"\n\n{_}\n\n")
        pass

    session = _maybe_await(
        session_service.get_session(
            app_name=app_name, user_id=user, session_id=session_id)
    )
    return session.state if session else {}
