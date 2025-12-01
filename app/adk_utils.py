"""
Utility to import the Google ADK package regardless of module name.

The pip package is `google-adk`. Some environments expose it as `google.adk`,
others as `adk`. This helper tries both to make the scaffold resilient.
"""

import importlib


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
