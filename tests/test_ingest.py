from __future__ import annotations

import unittest
from types import SimpleNamespace

from rag.ingest import count_docs, format_documents, load_pdfs


def document(text: str, source: str = "report.pdf", page: int | None = 1) -> SimpleNamespace:
    metadata = {"source": source}
    if page is not None:
        metadata["page"] = page
    return SimpleNamespace(page_content=text, metadata=metadata)


class IngestHelperTests(unittest.TestCase):
    def test_format_documents_adds_numbered_source_labels(self) -> None:
        rendered = format_documents(
            [document("first chunk", "a.pdf", 2), document("second chunk", "b.pdf", None)]
        )

        self.assertIn("[1] a.pdf, p.2", rendered)
        self.assertIn("[2] b.pdf", rendered)
        self.assertIn("second chunk", rendered)

    def test_format_documents_truncates_long_chunks(self) -> None:
        rendered = format_documents([document("x" * 200)], max_chars_per_doc=20)

        self.assertIn("x" * 20, rendered)
        self.assertNotIn("x" * 21, rendered)

    def test_format_documents_handles_empty_retrieval(self) -> None:
        self.assertEqual(format_documents([]), "No relevant context was found.")

    def test_count_docs_uses_public_chroma_api(self) -> None:
        store = SimpleNamespace(get=lambda **_: {"ids": ["a", "b", "c"]})

        self.assertEqual(count_docs(store), 3)

    def test_missing_pdf_is_rejected_before_importing_loaders(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_pdfs(["/tmp/definitely-missing.pdf"])

    def test_display_names_length_is_validated(self) -> None:
        with self.assertRaises(ValueError):
            load_pdfs([], display_names=["one.pdf"])


if __name__ == "__main__":
    unittest.main()
