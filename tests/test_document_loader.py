import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pypdf.errors import PdfStreamError

from src.document_loader import load_document, load_pdf


class DocumentLoaderTests(unittest.TestCase):
    def test_loads_text_document(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "notes.txt"
            path.write_text("A document fact.", encoding="utf-8")
            self.assertEqual(load_document(path), "A document fact.")

    def test_rejects_unsupported_file_extension(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "notes.md"
            path.write_text("Not a supported document.", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Only PDF and TXT"):
                load_document(path)

    def test_rejects_documents_over_the_size_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "notes.txt"
            path.write_bytes(b"x" * (1024 * 1024 + 1))
            with patch(
                "src.document_loader.MAX_DOCUMENT_SIZE_BYTES",
                1024 * 1024,
            ):
                with self.assertRaisesRegex(ValueError, "1 MB"):
                    load_document(path)

    def test_corrupt_pdf_raises_a_processing_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken.pdf"
            path.write_bytes(b"not a PDF")
            with self.assertRaises(PdfStreamError):
                load_document(path)

    def test_rejects_pdf_with_too_many_pages_before_extracting_text(self):
        with patch(
            "src.document_loader.PdfReader",
            return_value=type("Reader", (), {"pages": [None, None]})(),
        ):
            with self.assertRaisesRegex(ValueError, "1-page processing limit"):
                load_pdf("unused.pdf", max_pages=1)


if __name__ == "__main__":
    unittest.main()
