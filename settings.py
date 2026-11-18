"""Configuration loading and validation for the conversational RAG app.

All tunables live in ``config.yaml`` at the project root. Values are read once,
validated, and exposed as immutable dataclasses so the rest of the application
never has to guess which key exists or what type it has.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


class ConfigError(ValueError):
    """Raised when ``config.yaml`` contains an invalid or missing value."""


def _section(data: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = data.get(name, {})
    if not isinstance(value, Mapping):
        raise ConfigError(f"'{name}' must be a mapping, got {type(value).__name__}.")
    return value


def _as_str(section: Mapping[str, Any], key: str, default: str) -> str:
    value = section.get(key, default)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"'{key}' must be a non-empty string, got {value!r}.")
    return value.strip()


def _as_int(section: Mapping[str, Any], key: str, default: int, *, minimum: int = 1) -> int:
    value = section.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigError(f"'{key}' must be an integer >= {minimum}, got {value!r}.")
    return value


def _as_float(
    section: Mapping[str, Any],
    key: str,
    default: float,
    *,
    low: float,
    high: float,
) -> float:
    value = section.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"'{key}' must be a number, got {value!r}.")
    number = float(value)
    if not low <= number <= high:
        raise ConfigError(f"'{key}' must be between {low} and {high}, got {number}.")
    return number


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    model: str
    temperature: float


@dataclass(frozen=True)
class EmbeddingConfig:
    model: str


@dataclass(frozen=True)
class ChunkingConfig:
    chunk_size: int
    chunk_overlap: int


@dataclass(frozen=True)
class RetrievalConfig:
    search_type: str
    k: int
    fetch_k: int
    lambda_mult: float


@dataclass(frozen=True)
class RerankingConfig:
    enabled: bool
    model: str
    top_k: int


@dataclass(frozen=True)
class VectorStoreConfig:
    collection_name: str
    persist_directory: Path


@dataclass(frozen=True)
class Settings:
    """Validated application settings."""

    llm: LLMConfig
    embeddings: EmbeddingConfig
    chunking: ChunkingConfig
    retrieval: RetrievalConfig
    reranking: RerankingConfig
    vector_store: VectorStoreConfig

    @classmethod
    def from_yaml(cls, path: Path) -> "Settings":
        """Build settings from a YAML file, falling back to documented defaults."""
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ConfigError(f"Cannot read configuration file {path}: {exc}") from exc

        try:
            data = yaml.safe_load(raw) or {}
        except yaml.YAMLError as exc:
            raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc

        if not isinstance(data, Mapping):
            raise ConfigError("The configuration root must be a mapping.")

        llm = _section(data, "llm")
        embeddings = _section(data, "embeddings")
        chunking = _section(data, "chunking")
        retrieval = _section(data, "retrieval")
        reranking = _section(data, "reranking")
        vector_store = _section(data, "vector_store")

        chunk_size = _as_int(chunking, "chunk_size", 1000)
        chunk_overlap = _as_int(chunking, "chunk_overlap", 200, minimum=0)
        if chunk_overlap >= chunk_size:
            raise ConfigError("'chunk_overlap' must be smaller than 'chunk_size'.")

        search_type = _as_str(retrieval, "search_type", "mmr").lower()
        if search_type not in {"mmr", "similarity"}:
            raise ConfigError("'search_type' must be either 'mmr' or 'similarity'.")

        persist_raw = _as_str(vector_store, "persist_directory", "./chroma_db")
        persist_path = Path(persist_raw).expanduser()
        if not persist_path.is_absolute():
            persist_path = PROJECT_ROOT / persist_path

        settings = cls(
            llm=LLMConfig(
                provider=_as_str(llm, "provider", "groq"),
                model=_as_str(llm, "model", "llama-3.3-70b-versatile"),
                temperature=_as_float(llm, "temperature", 0.3, low=0.0, high=2.0),
            ),
            embeddings=EmbeddingConfig(
                model=_as_str(embeddings, "model", "sentence-transformers/all-MiniLM-L6-v2")
            ),
            chunking=ChunkingConfig(chunk_size=chunk_size, chunk_overlap=chunk_overlap),
            retrieval=RetrievalConfig(
                search_type=search_type,
                k=_as_int(retrieval, "k", 4),
                fetch_k=_as_int(retrieval, "fetch_k", 20),
                lambda_mult=_as_float(retrieval, "lambda_mult", 0.5, low=0.0, high=1.0),
            ),
            reranking=RerankingConfig(
                enabled=reranking.get("enabled", False) is True,
                model=_as_str(
                    reranking,
                    "model",
                    "cross-encoder/ms-marco-MiniLM-L-6-v2",
                ),
                top_k=_as_int(reranking, "top_k", 4),
            ),
            vector_store=VectorStoreConfig(
                collection_name=_as_str(vector_store, "collection_name", "pdf_collection"),
                persist_directory=persist_path,
            ),
        )
        return _apply_env_overrides(settings)


def _env_int(name: str, current: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return current
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"'{name}' must be an integer, got {raw!r}.") from exc


def _env_float(name: str, current: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return current
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"'{name}' must be a number, got {raw!r}.") from exc


def _apply_env_overrides(settings: Settings) -> Settings:
    """Allow a few common values to be overridden without editing the YAML."""
    model = os.getenv("RAG_LLM_MODEL")
    embedding_model = os.getenv("RAG_EMBEDDINGS_MODEL")
    temperature = os.getenv("RAG_TEMPERATURE")
    k = os.getenv("RAG_RETRIEVAL_K")

    if not any((model, embedding_model, temperature, k)):
        return settings

    llm = replace(
        settings.llm,
        model=model or settings.llm.model,
        temperature=_env_float("RAG_TEMPERATURE", settings.llm.temperature),
    )
    embeddings = replace(
        settings.embeddings,
        model=embedding_model or settings.embeddings.model,
    )
    retrieval = replace(settings.retrieval, k=_env_int("RAG_RETRIEVAL_K", settings.retrieval.k))
    return replace(settings, llm=llm, embeddings=embeddings, retrieval=retrieval)


@lru_cache(maxsize=4)
def load_settings(path: str | Path | None = None) -> Settings:
    """Return cached settings. ``RAG_CONFIG_PATH`` can select a different file."""
    config_path = Path(path) if path is not None else Path(
        os.getenv("RAG_CONFIG_PATH", str(DEFAULT_CONFIG_PATH))
    ).expanduser()
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    return Settings.from_yaml(config_path)


def get_settings() -> Settings:
    """Convenience accessor used by the runtime modules."""
    return load_settings()


def reload_settings() -> None:
    """Drop the cache, primarily useful in tests and long-running dev sessions."""
    load_settings.cache_clear()
