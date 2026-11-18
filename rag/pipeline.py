"""LangChain LCEL pipeline: history-aware retrieval followed by grounded answering."""

from __future__ import annotations

import logging
from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import Any

from rag.ingest import format_documents
from settings import ConfigError, Settings, get_settings

logger = logging.getLogger(__name__)

CONTEXTUALIZE_SYSTEM_PROMPT = (
    "Given a chat history and the latest user question, which may refer to "
    "something said earlier, rewrite the question so it is fully self-contained. "
    "Do not answer it. Keep the original wording when no rewrite is needed."
)

ANSWER_SYSTEM_PROMPT = (
    "You answer questions using only the retrieved context below.\n"
    "Each context block starts with a label such as [1] report.pdf, p.3.\n"
    "Cite the label of every claim you make, for example: '... [1]'.\n"
    "If the context does not contain the answer, say: "
    "\"I couldn't find that in the uploaded documents.\"\n"
    "Be concise: at most five sentences.\n\n"
    "{context}"
)


@dataclass(frozen=True)
class RagPipeline:
    """Everything the engine needs to answer questions for a session."""

    chain: Any
    history_store: MutableMapping[str, Any]
    settings: Settings

    def history(self, session_id: str) -> Any:
        return self.history_store.get(session_id)

    def clear_history(self, session_id: str) -> None:
        history = self.history_store.get(session_id)
        if history is not None:
            history.clear()


def build_retriever(vector_store: Any, *, settings: Settings | None = None) -> Any:
    """Create a retriever from the configured search strategy."""
    config = settings or get_settings()
    retrieval = config.retrieval

    if retrieval.search_type == "mmr":
        return vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": retrieval.k,
                "fetch_k": retrieval.fetch_k,
                "lambda_mult": retrieval.lambda_mult,
            },
        )
    if retrieval.search_type == "similarity":
        return vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": retrieval.k},
        )
    raise ConfigError(f"Unsupported search_type: {retrieval.search_type!r}.")


class RerankingRetriever:
    """Optional second stage that rescores candidates with a cross-encoder."""

    def __init__(
        self,
        base_retriever: Any,
        model_name: str,
        top_k: int,
        *,
        model: Any = None,
    ) -> None:
        self.base_retriever = base_retriever
        self.model_name = model_name
        self.top_k = top_k
        self._model = model

    def invoke(self, question: str) -> list[Any]:
        documents = list(self.base_retriever.invoke(question))
        if not documents:
            return []

        scores = self._get_model().predict(
            [(question, document.page_content) for document in documents]
        )
        ranked = sorted(
            zip(documents, (float(score) for score in scores)),
            key=lambda pair: pair[1],
            reverse=True,
        )
        return [document for document, _ in ranked[: self.top_k]]

    def _get_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            logger.info("Loading reranker %s", self.model_name)
            self._model = CrossEncoder(self.model_name)
        return self._model


def make_history_store() -> tuple[dict[str, Any], Any]:
    """Create an isolated, in-memory history registry for one pipeline."""
    from langchain_core.chat_history import InMemoryChatMessageHistory

    store: dict[str, Any] = {}

    def get_session_history(session_id: str) -> Any:
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]

    return store, get_session_history


def create_rag_pipeline(
    vector_store: Any,
    api_key: str,
    *,
    settings: Settings | None = None,
) -> RagPipeline:
    """Build the full history-aware chain.

    The key is passed straight to the provider client and is never persisted.
    """
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.runnables.history import RunnableWithMessageHistory
    from langchain_groq import ChatGroq

    if not api_key or not api_key.strip():
        raise ValueError("A Groq API key is required to build the RAG pipeline.")

    config = settings or get_settings()
    llm = ChatGroq(
        model=config.llm.model,
        temperature=config.llm.temperature,
        groq_api_key=api_key.strip(),
    )

    base_retriever = build_retriever(vector_store, settings=config)
    retriever: Any = base_retriever
    if config.reranking.enabled:
        retriever = RerankingRetriever(
            base_retriever,
            config.reranking.model,
            config.reranking.top_k,
        )

    history_store, get_session_history = make_history_store()

    contextualize_chain = (
        ChatPromptTemplate.from_messages(
            [
                ("system", CONTEXTUALIZE_SYSTEM_PROMPT),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        | llm
        | StrOutputParser()
    )

    answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", ANSWER_SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    def standalone_question(payload: dict[str, Any]) -> str:
        history = payload.get("chat_history") or []
        if not history:
            return str(payload["input"])
        rewritten = contextualize_chain.invoke(
            {"input": payload["input"], "chat_history": history}
        )
        return str(rewritten).strip() or str(payload["input"])

    def retrieved_context(payload: dict[str, Any]) -> str:
        documents = retriever.invoke(standalone_question(payload))
        return format_documents(documents)

    rag_chain = (
        {
            "context": retrieved_context,
            "input": lambda payload: payload["input"],
            "chat_history": lambda payload: payload.get("chat_history", []),
        }
        | answer_prompt
        | llm
        | StrOutputParser()
    )

    chain_with_history = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )
    return RagPipeline(chain=chain_with_history, history_store=history_store, settings=config)
