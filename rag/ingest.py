"""PDF -> chunks -> embeddings -> Chroma ingestion pipeline.

Heavy third-party imports are deliberately kept inside functions so the helper
logic can be unit-tested without downloading embedding models.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from settings import Settings, get_settings

logger = logging.getLogger(__name__)

PathLike = str | Path


def load_pdfs(
    paths: Sequence[PathLike],
    *,
    display_names: Sequence[str] | None = None,
) -> list[Any]:
    """Load PDF pages and normalize their metadata.

    ``display_names`` keeps the original upload names in citations even though
    the files are read from temporary paths.
    """
    file_paths = [Path(path) for path in paths]
    if display_names is not None and len(display_names) != len(file_paths):
        raise ValueError("display_names must have the same length as paths.")

    for path in file_paths:
        if not path.is_file():
            raise FileNotFoundError(f"PDF not found: {path}")
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a .pdf file, got: {path.name}")

    from langchain_community.document_loaders import PyPDFLoader

    documents: list[Any] = []
    for index, path in enumerate(file_paths):
        pages = PyPDFLoader(str(path)).load()
        label = display_names[index] if display_names else path.name
        for document in pages:
            document.metadata["source"] = Path(label).name
            try:
                page_number = int(document.metadata.get("page", 0))
            except (TypeError, ValueError):
                page_number = 0
            document.metadata["page"] = page_number + 1
            documents.append(document)
    return documents


def split_docs(documents: Sequence[Any], *, settings: Settings | None = None) -> list[Any]:
    """Split loaded pages into overlapping chunks using the configured sizes."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    config = settings or get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunking.chunk_size,
        chunk_overlap=config.chunking.chunk_overlap,
        add_start_index=True,
    )
    return splitter.split_documents(list(documents))


def get_embeddings(*, settings: Settings | None = None) -> Any:
    """Return the local HuggingFace embedding model (no API key required)."""
    from langchain_huggingface import HuggingFaceEmbeddings

    config = settings or get_settings()
    return HuggingFaceEmbeddings(
        model_name=config.embeddings.model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def create_vector_store(
    chunks: Sequence[Any],
    *,
    collection_name: str | None = None,
    settings: Settings | None = None,
    replace: bool = True,
) -> Any:
    """Index chunks in Chroma.

    Only the named collection is replaced. The rest of the persisted directory
    is left untouched, so unrelated collections are never deleted by accident.
    """
    from langchain_chroma import Chroma

    config = settings or get_settings()
    name = collection_name or config.vector_store.collection_name
    persist_directory = config.vector_store.persist_directory
    persist_directory.mkdir(parents=True, exist_ok=True)

    embeddings = get_embeddings(settings=config)
    if replace:
        existing = Chroma(
            collection_name=name,
            embedding_function=embeddings,
            persist_directory=str(persist_directory),
        )
        try:
            existing.delete_collection()
        except Exception as exc:  # noqa: BLE001 - Chroma raises varied errors
            logger.warning("Could not drop existing collection %s: %s", name, exc)

    return Chroma.from_documents(
        documents=list(chunks),
        embedding=embeddings,
        collection_name=name,
        persist_directory=str(persist_directory),
    )


def get_vector_store(
    *,
    collection_name: str | None = None,
    settings: Settings | None = None,
) -> Any:
    """Open an existing collection (Chroma creates it if it is absent)."""
    from langchain_chroma import Chroma

    config = settings or get_settings()
    return Chroma(
        collection_name=collection_name or config.vector_store.collection_name,
        embedding_function=get_embeddings(settings=config),
        persist_directory=str(config.vector_store.persist_directory),
    )


def count_docs(vector_store: Any) -> int:
    """Return the number of indexed chunks using Chroma's public API."""
    return len(vector_store.get(include=[])["ids"])


def format_documents(documents: Sequence[Any], *, max_chars_per_doc: int = 1200) -> str:
    """Render retrieved chunks with numbered source labels for grounding."""
    blocks: list[str] = []
    for index, document in enumerate(documents, start=1):
        metadata = getattr(document, "metadata", None) or {}
        source = str(metadata.get("source", "unknown"))
        page = metadata.get("page")
        label = f"[{index}] {source}"
        if page not in (None, ""):
            label += f", p.{page}"

        text = str(getattr(document, "page_content", "")).strip()
        if len(text) > max_chars_per_doc:
            text = text[:max_chars_per_doc].rstrip() + " …"
        blocks.append(f"{label}\n{text}")

    return "\n\n".join(blocks) or "No relevant context was found."
