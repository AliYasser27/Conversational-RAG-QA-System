"""Small deterministic engine kept as a reference implementation.

It satisfies the same :class:`~engines.base.ChatEngine` protocol as the RAG
engine and is useful for demoing the Strategy pattern without an API key.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping

from .base import ChatEngine, Message

UNKNOWN_ANSWER = "Sorry, I don't have an answer for that."


class SimpleFAQEngine(ChatEngine):
    """Lookup-based engine with per-session history."""

    def __init__(self, faq_map: Mapping[str, str] | None = None) -> None:
        self.faq_map = {
            str(question).strip().lower(): str(answer)
            for question, answer in (faq_map or {}).items()
        }
        self.histories: defaultdict[str, list[Message]] = defaultdict(list)

    def answer(self, session_id: str, question: str) -> str:
        normalized = question.strip().lower()
        answer = self.faq_map.get(normalized, UNKNOWN_ANSWER)

        self.histories[session_id].append(("user", question))
        self.histories[session_id].append(("assistant", answer))
        return answer

    def get_history(self, session_id: str) -> list[Message]:
        return list(self.histories.get(session_id, []))

    def clear_history(self, session_id: str) -> None:
        self.histories.pop(session_id, None)
