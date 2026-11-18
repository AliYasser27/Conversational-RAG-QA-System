"""Interface shared by every chat engine.

The Streamlit layer depends only on this protocol, so the FAQ demo and the
RAG engine are interchangeable without changing any UI code.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

Message = tuple[str, str]


class ChatEngine(Protocol):
    """Minimal contract for a session-aware conversational engine."""

    def answer(self, session_id: str, question: str) -> str:
        """Return an answer for ``question`` within ``session_id``."""
        ...

    def get_history(self, session_id: str) -> Sequence[Message]:
        """Return the transcript for ``session_id`` as ``(role, text)`` pairs."""
        ...

    def clear_history(self, session_id: str) -> None:
        """Forget all messages belonging to ``session_id``."""
        ...
