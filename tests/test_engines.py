from __future__ import annotations

import unittest
from types import SimpleNamespace

from engines.rag_engine import RagEngine, _text_from
from engines.simple_faq import UNKNOWN_ANSWER, SimpleFAQEngine


class FakeHistory:
    def __init__(self, messages=None) -> None:
        self.messages = list(messages or [])

    def clear(self) -> None:
        self.messages = []


class FakeChain:
    def __init__(self, result: object = "grounded answer") -> None:
        self.result = result
        self.calls: list[tuple[dict, dict | None]] = []

    def invoke(self, payload: dict, config: dict | None = None) -> object:
        self.calls.append((payload, config))
        return self.result


class SimpleFAQEngineTests(unittest.TestCase):
    def test_lookup_is_case_insensitive_and_history_is_recorded(self) -> None:
        engine = SimpleFAQEngine({"What is RAG?": "Grounded generation."})

        answer = engine.answer("s1", "  what is rag?  ")

        self.assertEqual(answer, "Grounded generation.")
        self.assertEqual(
            engine.get_history("s1"),
            [("user", "  what is rag?  "), ("assistant", "Grounded generation.")],
        )
        self.assertEqual(engine.get_history("s2"), [])

    def test_unknown_question_uses_default_answer(self) -> None:
        engine = SimpleFAQEngine()

        self.assertEqual(engine.answer("s1", "anything"), UNKNOWN_ANSWER)
        engine.clear_history("s1")
        self.assertEqual(engine.get_history("s1"), [])


class RagEngineTests(unittest.TestCase):
    def test_answer_passes_session_id_and_returns_text(self) -> None:
        chain = FakeChain(SimpleNamespace(content="  answer text  "))
        store = {"s1": FakeHistory()}
        engine = RagEngine(chain, store)

        self.assertEqual(engine.answer("s1", "  why?  "), "answer text")
        payload, config = chain.calls[0]
        self.assertEqual(payload, {"input": "why?"})
        self.assertEqual(config, {"configurable": {"session_id": "s1"}})

    def test_empty_question_is_rejected(self) -> None:
        engine = RagEngine(FakeChain(), {})

        with self.assertRaises(ValueError):
            engine.answer("s1", "   ")

    def test_history_roles_are_normalized(self) -> None:
        history = FakeHistory(
            [
                SimpleNamespace(type="human", content="hello"),
                SimpleNamespace(type="ai", content="hi"),
            ]
        )
        engine = RagEngine(FakeChain(), {"s1": history})

        self.assertEqual(engine.get_history("s1"), [("user", "hello"), ("assistant", "hi")])
        engine.clear_history("s1")
        self.assertEqual(engine.get_history("s1"), [])

    def test_output_coercion_handles_common_shapes(self) -> None:
        self.assertEqual(_text_from(SimpleNamespace(content=[{"text": "a"}, "b"])), "ab")
        self.assertEqual(_text_from({"answer": " from dict "}), "from dict")


if __name__ == "__main__":
    unittest.main()
