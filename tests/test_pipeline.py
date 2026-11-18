from __future__ import annotations

import unittest
from dataclasses import replace
from types import SimpleNamespace

from rag.pipeline import (
    ANSWER_SYSTEM_PROMPT,
    CONTEXTUALIZE_SYSTEM_PROMPT,
    RagPipeline,
    RerankingRetriever,
    build_retriever,
)
from settings import ConfigError, load_settings


class FakeRetriever:
    def __init__(self, documents=None) -> None:
        self.documents = list(documents or [])
        self.questions: list[str] = []

    def invoke(self, question: str) -> list:
        self.questions.append(question)
        return self.documents


class FakeStore:
    def __init__(self, documents=None) -> None:
        self.retriever = FakeRetriever(documents)
        self.kwargs: dict = {}

    def as_retriever(self, **kwargs) -> FakeRetriever:
        self.kwargs = kwargs
        return self.retriever


class FakeReranker:
    def predict(self, pairs) -> list[float]:
        return [0.1, 0.9, 0.5]


class PipelineTests(unittest.TestCase):
    def test_mmr_kwargs_come_from_config(self) -> None:
        store = FakeStore()
        settings = load_settings()

        build_retriever(store, settings=settings)

        self.assertEqual(store.kwargs["search_type"], "mmr")
        self.assertEqual(store.kwargs["search_kwargs"]["k"], settings.retrieval.k)
        self.assertEqual(store.kwargs["search_kwargs"]["fetch_k"], settings.retrieval.fetch_k)

    def test_similarity_search_uses_only_k(self) -> None:
        store = FakeStore()
        base = load_settings()
        settings = replace(base, retrieval=replace(base.retrieval, search_type="similarity"))

        build_retriever(store, settings=settings)

        self.assertEqual(store.kwargs["search_type"], "similarity")
        self.assertEqual(store.kwargs["search_kwargs"], {"k": settings.retrieval.k})

    def test_invalid_search_type_raises(self) -> None:
        base = load_settings()
        settings = replace(base, retrieval=replace(base.retrieval, search_type="hybrid"))

        with self.assertRaises(ConfigError):
            build_retriever(FakeStore(), settings=settings)

    def test_reranker_sorts_by_score_and_limits_results(self) -> None:
        documents = [SimpleNamespace(page_content=name) for name in ("low", "high", "mid")]
        retriever = RerankingRetriever(
            FakeRetriever(documents), "dummy-model", 2, model=FakeReranker()
        )

        result = retriever.invoke("question")

        self.assertEqual([doc.page_content for doc in result], ["high", "mid"])

    def test_prompts_require_grounding_and_rewriting(self) -> None:
        self.assertIn("cite", ANSWER_SYSTEM_PROMPT.lower())
        self.assertIn("{context}", ANSWER_SYSTEM_PROMPT)
        self.assertIn("self-contained", CONTEXTUALIZE_SYSTEM_PROMPT.lower())

    def test_pipeline_history_helpers(self) -> None:
        class History:
            def __init__(self) -> None:
                self.messages = ["old"]

            def clear(self) -> None:
                self.messages = []

        history = History()
        pipeline = RagPipeline(
            chain=object(), history_store={"s1": history}, settings=load_settings()
        )

        self.assertIs(pipeline.history("s1"), history)
        pipeline.clear_history("s1")
        self.assertEqual(history.messages, [])


if __name__ == "__main__":
    unittest.main()
