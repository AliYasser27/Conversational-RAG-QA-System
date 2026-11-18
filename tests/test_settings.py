from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from settings import DEFAULT_CONFIG_PATH, ConfigError, Settings, load_settings


class SettingsTests(unittest.TestCase):
    def test_repository_config_loads_and_validates(self) -> None:
        settings = load_settings()

        self.assertEqual(settings.chunking.chunk_size, 1000)
        self.assertEqual(settings.chunking.chunk_overlap, 200)
        self.assertEqual(settings.retrieval.k, 4)
        self.assertEqual(settings.retrieval.search_type, "mmr")
        self.assertTrue(settings.vector_store.persist_directory.is_absolute())

    def test_overlap_must_be_smaller_than_chunk_size(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text("chunking:\n  chunk_size: 100\n  chunk_overlap: 100\n")

            with self.assertRaises(ConfigError):
                Settings.from_yaml(path)

    def test_unknown_search_type_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text("retrieval:\n  search_type: hybrid\n")

            with self.assertRaises(ConfigError):
                Settings.from_yaml(path)

    def test_missing_file_reports_its_path(self) -> None:
        with self.assertRaisesRegex(ConfigError, "missing.yaml"):
            Settings.from_yaml(Path("/tmp/missing.yaml"))

    def test_environment_can_override_model(self) -> None:
        with patch.dict(os.environ, {"RAG_LLM_MODEL": "test-model"}, clear=False):
            settings = Settings.from_yaml(DEFAULT_CONFIG_PATH)

        self.assertEqual(settings.llm.model, "test-model")


if __name__ == "__main__":
    unittest.main()
