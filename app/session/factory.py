"""
Session and memory wiring for ADK.

This is a thin wrapper around adk-python to create a session per user/upload,
attach vector memory, and provide a place to stash recent state.
"""

from pathlib import Path
from typing import Optional

from app.config import settings

# Prefer documented import path; fall back to legacy name if needed.
try:
    from google.adk.session import Session  # type: ignore
except ImportError:  # pragma: no cover
    try:
        from adk import Session  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "google-adk is required (expected Session in google.adk.session). "
            "Install with `pip install google-adk`."
        ) from exc


def get_session(session_id: str, working_dir: Optional[Path] = None):
    """
    Create or restore an ADK session with configured memory stores.

    Args:
        session_id: Unique identifier for a user/upload scope.
        working_dir: Optional override for where to persist state.
    """
    base_dir = working_dir or settings.data_dir
    base_dir.mkdir(parents=True, exist_ok=True)
    # NOTE: adjust once specific ADK memory APIs are known.
    session = Session(id=session_id, storage_path=str(base_dir / f"{session_id}.json"))
    return session
