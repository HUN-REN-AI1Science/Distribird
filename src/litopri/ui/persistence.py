"""Browser-side session persistence via localStorage.

Stores sidebar settings and login status as a JSON blob so the left
panel survives page reloads and sleep/wake cycles.

Security note: API keys stored in localStorage are accessible to any JS
running on the same origin. Acceptable for a single-user local tool.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import streamlit as st

logger = logging.getLogger(__name__)

STORAGE_KEY = "litopri_state"

# Keys persisted as-is — sidebar (left panel) settings only
_SIMPLE_KEYS: list[str] = [
    "authenticated",
    # Sidebar toggles
    "use_s2",
    "use_openalex",
    "use_deep",
    "web_search",
    "override_toggle",
    # Search settings
    "max_q",
    "max_p",
    # Sidebar text inputs (required fields)
    "llm_url_req",
    "llm_key_req",
    "llm_model_req",
    "s2_key_req",
    "dr_url_req",
    "dr_key_req",
    "dr_model_req",
    # Sidebar text inputs (override fields)
    "llm_url_ov",
    "llm_key_ov",
    "llm_model_ov",
    "s2_key_ov",
    "dr_url_ov",
    "dr_key_ov",
    "dr_model_ov",
    "oa_email_ov",
]


def hydrate_session_state() -> object:
    """Restore session state from localStorage on first render.

    Must be called at the top of ``main()`` before any widgets render.
    Returns the ``LocalStorage`` instance for reuse in ``save_session_state``.
    """
    from streamlit_local_storage import LocalStorage  # type: ignore[import-untyped]

    ls = LocalStorage()

    if st.session_state.get("_hydrated"):
        return ls

    try:
        stored = ls.getAll()
        if not stored or STORAGE_KEY not in stored:
            st.session_state["_hydrated"] = True
            return ls

        raw = stored[STORAGE_KEY]
        blob: dict[str, Any] = json.loads(raw) if isinstance(raw, str) else raw

        # Restore simple keys
        for key in _SIMPLE_KEYS:
            if key in blob:
                st.session_state[key] = blob[key]

    except Exception:
        logger.warning("Failed to hydrate from localStorage", exc_info=True)

    st.session_state["_hydrated"] = True
    return ls


def save_session_state(ls: object) -> None:
    """Persist current session state to localStorage.

    Called at the end of ``main()`` after all widgets have rendered.
    """
    try:
        blob: dict[str, Any] = {}

        for key in _SIMPLE_KEYS:
            if key in st.session_state:
                blob[key] = st.session_state[key]

        ls.setItem(STORAGE_KEY, json.dumps(blob, default=str))  # type: ignore[attr-defined]
    except Exception:
        logger.debug("Failed to save to localStorage", exc_info=True)


def clear_persisted_state(ls: object) -> None:
    """Remove all persisted state from localStorage."""
    try:
        ls.deleteAll()  # type: ignore[attr-defined]
    except Exception:
        logger.debug("Failed to clear localStorage", exc_info=True)
