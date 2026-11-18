"""Adapter between the LangChain RAG chain and the UI-facing ChatEngine."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from typing import Any

from .base import ChatEngine, Message

USER_ROLES = {"human", "user"}


def _text_from(result: Any) -> str:
    """Normalize common LangChain output shapes into a plain string."""
    if isinstance(result, str):
        return result.strip()

    content = getattr(result, "content", result)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, Mapping) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "".join(parts).strip()
    if isinstance(result, Mapping):
        for key in ("answer", "output_text", "output"):
            value = result.get(key)
            if isinstance(value, str):
                return value.strip()

    raise TypeError(f"Unsupported chain output type: {type(result).__name__}.")


class RagEngine(ChatEngine):
    """Executes a history-aware chain and exposes per-session transcripts."""

    def __init__(self, chain: Any, history_store: MutableMapping[str, Any]) -> None:
        self.chain = chain
        self.history_store = history_store

    def answer(self, session_id: str, question: str) -> str:
        question = question.strip()
        if not question:
            raise ValueError("Question must not be empty.")

        result = self.chain.invoke(
            {"input": question},
            config={"configurable": {"session_id": session_id}},
        )
        return _text_from(result)

    def get_history(self, session_id: str) -> list[Message]:
        history = self.history_store.get(session_id)
        if history is None:
            return []

        messages: list[Message] = []
        for message in history.messages:
            role = "user" if getattr(message, "type", "") in USER_ROLES else "assistant"
            messages.append((role, str(message.content)))
        return messages

    def clear_history(self, session_id: str) -> None:
        history = self.history_store.get(session_id)
        if history is not None:
            history.clear()
