"""Streamlit UI for conversational question answering over uploaded PDFs.

The UI only knows about the ChatEngine protocol. Swapping the FAQ demo for the
RAG engine (or vice versa) does not change any presentation code.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import streamlit as st

from engines.base import ChatEngine
from engines.rag_engine import RagEngine
from engines.simple_faq import SimpleFAQEngine
from rag.ingest import count_docs, create_vector_store, load_pdfs, split_docs
from rag.pipeline import create_rag_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_UPLOAD_MB = 25

DEFAULT_FAQ = {
    "what is rag?": (
        "Retrieval-Augmented Generation grounds an LLM answer in documents "
        "retrieved at query time."
    ),
    "which models does this app use?": (
        "Local HuggingFace embeddings for retrieval and a Groq-hosted Llama model "
        "for answering."
    ),
    "how many sessions are supported?": "As many as you like; each keeps its own transcript.",
}


class Chatbot:
    """Facade used by the UI; delegates to an injected engine."""

    def __init__(self, engine: ChatEngine) -> None:
        self.engine = engine

    def ask(self, session_id: str, question: str) -> str:
        return self.engine.answer(session_id, question)

    def history(self, session_id: str):
        return self.engine.get_history(session_id)

    def clear_history(self, session_id: str) -> None:
        self.engine.clear_history(session_id)


def init_state() -> None:
    """Populate Streamlit session state on the first run."""
    defaults = {
        "sessions": {"session-1": {"name": "Chat 1", "messages": []}},
        "current_session_id": "session-1",
        "session_counter": 1,
        "messages": [],
        "chatbot": None,
        "documents_ready": False,
        "bound_api_key": None,
        "indexed_chunks": 0,
        "engine_mode": "rag",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def current_session() -> dict:
    return st.session_state.sessions[st.session_state.current_session_id]


def persist_messages() -> None:
    current_session()["messages"] = list(st.session_state.messages)


def activate_session(session_id: str) -> None:
    persist_messages()
    st.session_state.current_session_id = session_id
    st.session_state.messages = list(st.session_state.sessions[session_id]["messages"])


def reset_engine() -> None:
    st.session_state.chatbot = None
    st.session_state.documents_ready = False
    st.session_state.bound_api_key = None
    st.session_state.indexed_chunks = 0


def process_uploads(uploads, api_key: str) -> None:
    """Validate uploads, rebuild the index, and install a fresh RAG engine."""
    if not api_key.strip():
        st.error("Add your Groq API key before processing PDFs.")
        return
    if not uploads:
        st.error("Choose at least one PDF first.")
        return

    try:
        with tempfile.TemporaryDirectory(prefix="rag_uploads_") as temporary_dir:
            paths: list[Path] = []
            names: list[str] = []
            for upload in uploads:
                if not upload.name.lower().endswith(".pdf"):
                    raise ValueError(f"Unsupported file type: {upload.name}")
                if upload.size > MAX_UPLOAD_MB * 1024 * 1024:
                    raise ValueError(
                        f"{upload.name} exceeds the {MAX_UPLOAD_MB} MB upload limit."
                    )
                path = Path(temporary_dir) / upload.name
                path.write_bytes(upload.getbuffer())
                paths.append(path)
                names.append(upload.name)

            with st.spinner("Extracting text and building embeddings…"):
                documents = load_pdfs(paths, display_names=names)
                chunks = split_docs(documents)
                if not chunks:
                    raise ValueError("No extractable text was found in the uploaded PDFs.")
                vector_store = create_vector_store(chunks)
                pipeline = create_rag_pipeline(vector_store, api_key)

        st.session_state.chatbot = Chatbot(RagEngine(pipeline.chain, pipeline.history_store))
        st.session_state.documents_ready = True
        st.session_state.bound_api_key = api_key
        st.session_state.indexed_chunks = count_docs(vector_store)

        # Old transcripts referred to the previous index, so start them clean.
        for session in st.session_state.sessions.values():
            session["messages"] = []
        st.session_state.messages = []

        st.success(f"Indexed {st.session_state.indexed_chunks} chunks. Chat history was reset.")
    except Exception as exc:
        logger.exception("PDF ingestion failed")
        st.error(f"Could not index the PDFs: {exc}")


def render_sessions_sidebar() -> None:
    st.title("💬 Sessions")
    st.caption("Each session keeps an independent transcript.")

    for session_id, session in list(st.session_state.sessions.items()):
        is_current = session_id == st.session_state.current_session_id
        columns = st.columns([4, 1])

        with columns[0]:
            name = session["name"]
            label = f"{'●' if is_current else '○'} {name[:18]}"
            if st.button(label, key=f"select_{session_id}", use_container_width=True):
                if not is_current:
                    activate_session(session_id)
                    st.rerun()

        with columns[1]:
            if len(st.session_state.sessions) > 1 and st.button(
                "×", key=f"delete_{session_id}", help=f"Delete {name}"
            ):
                persist_messages()
                del st.session_state.sessions[session_id]
                if is_current:
                    activate_session(next(iter(st.session_state.sessions)))
                st.rerun()

    if st.button("➕ New chat", use_container_width=True):
        persist_messages()
        st.session_state.session_counter += 1
        new_id = f"session-{st.session_state.session_counter}"
        st.session_state.sessions[new_id] = {
            "name": f"Chat {st.session_state.session_counter}",
            "messages": [],
        }
        activate_session(new_id)
        st.rerun()


def render_settings_sidebar() -> None:
    st.divider()
    st.title("⚙️ Settings")

    mode_label = st.radio(
        "Engine",
        ["RAG over PDFs", "FAQ demo (no key)"],
        index=0 if st.session_state.engine_mode == "rag" else 1,
        horizontal=True,
    )
    requested_mode = "rag" if mode_label.startswith("RAG") else "faq"
    if requested_mode != st.session_state.engine_mode:
        st.session_state.engine_mode = requested_mode
        reset_engine()
        st.rerun()

    if st.session_state.engine_mode == "faq":
        st.session_state.chatbot = Chatbot(SimpleFAQEngine(DEFAULT_FAQ))
        st.caption("Deterministic FAQ mode — demonstrates the swappable engine interface.")
        return

    api_key = st.text_input(
        "Groq API key",
        value=os.getenv("GROQ_API_KEY", ""),
        type="password",
        key="groq_api_key",
        help="Held in memory only. You can also set GROQ_API_KEY locally.",
    )
    if st.session_state.bound_api_key and api_key != st.session_state.bound_api_key:
        reset_engine()
        st.warning("API key changed — process the PDFs again to rebuild the engine.")
    st.markdown("[Get a key](https://console.groq.com/keys)")

    st.divider()
    st.subheader("📄 Documents")
    uploads = st.file_uploader(
        "Upload PDFs",
        accept_multiple_files=True,
        type=["pdf"],
        help=f"Maximum {MAX_UPLOAD_MB} MB per file.",
    )

    if st.button("🚀 Process PDFs", type="primary", use_container_width=True):
        process_uploads(uploads, api_key)

    if st.session_state.documents_ready:
        st.success(f"Index ready · {st.session_state.indexed_chunks} chunks")
    else:
        st.info("Upload PDFs and click **Process PDFs**.")


def render_chat() -> None:
    session_id = st.session_state.current_session_id
    st.caption(
        f"Session: **{current_session()['name']}** · "
        f"engine: {st.session_state.engine_mode}"
    )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    chatbot = st.session_state.chatbot
    if chatbot is None:
        st.info("Configure an engine in the sidebar to start chatting.")
        return

    placeholder = (
        "Ask about the uploaded PDFs…"
        if st.session_state.engine_mode == "rag"
        else "Ask the FAQ demo…"
    )
    question = st.chat_input(placeholder)
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                answer = chatbot.ask(session_id, question)
            except Exception as exc:
                logger.exception("Answer generation failed")
                answer = f"Sorry, I couldn't generate an answer. ({exc})"
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    persist_messages()


def main() -> None:
    st.set_page_config(page_title="Conversational RAG Q&A", page_icon="📚", layout="wide")
    init_state()

    with st.sidebar:
        render_sessions_sidebar()
        render_settings_sidebar()
        st.divider()
        if st.button("🗑️ Clear this chat", use_container_width=True):
            st.session_state.messages = []
            current_session()["messages"] = []
            if st.session_state.chatbot is not None:
                st.session_state.chatbot.clear_history(st.session_state.current_session_id)
            st.rerun()

    st.title("Conversational RAG Q&A")
    st.caption("Upload PDFs, ask questions, and follow up naturally across turns.")
    render_chat()

    with st.expander("How this works"):
        st.markdown(
            "1. PDFs are parsed page by page and split into overlapping chunks.\n"
            "2. Local HuggingFace embeddings are stored in a Chroma collection.\n"
            "3. A follow-up question is rewritten using the session history.\n"
            "4. MMR retrieval (plus optional reranking) selects the relevant chunks.\n"
            "5. The answer is generated from those chunks with source labels."
        )


if __name__ == "__main__":
    main()
