import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from src.document_tasks import (
    generate_quiz,
    get_document_chunks,
    parse_quiz,
    summarize_document,
    validate_quiz,
)
from src.knowledge_base import KnowledgeBase
from src.vector_store import VectorStore
from src.embeddings import MODEL_NAME


class DocumentTaskTests(unittest.TestCase):
    def setUp(self):
        self.knowledge_base = SimpleNamespace(
            vector_store=SimpleNamespace(
                text_chunks=[
                    "Naila is a civil engineer. She has five years of experience.",
                    "Project management is one of her skills.",
                    "This belongs to Naila-CV-archive.txt, not Naila-CV.txt.",
                ],
                sources=[
                    "Naila-CV.txt",
                    "Naila-CV.txt",
                    "Naila-CV-archive.txt",
                ],
            )
        )

    def test_document_selection_matches_exact_filename(self):
        chunks = get_document_chunks("Naila-CV.txt", self.knowledge_base)
        self.assertEqual(len(chunks), 2)
        self.assertNotIn("archive", "\n".join(chunks))

    def test_summary_is_extracted_from_selected_document_only(self):
        chunks = get_document_chunks("Naila-CV.txt", self.knowledge_base)
        source_text = " ".join(chunks)
        summary = summarize_document("Naila-CV.txt", self.knowledge_base)
        self.assertTrue(summary)
        self.assertNotIn("archive", summary)
        for sentence in summary.split("\n\n"):
            self.assertIn(sentence, source_text)

    def test_parse_quiz_accepts_json_fences_and_rejects_invalid_json(self):
        payload = [{"question": "Q?"}]
        self.assertEqual(
            parse_quiz(f"```json\n{json.dumps(payload)}\n```"),
            payload,
        )
        self.assertEqual(
            parse_quiz('[{"question":"complete"},{"question":"unfinished"'),
            [{"question": "complete"}],
        )
        self.assertEqual(parse_quiz("not json"), [])

    def test_quiz_validation_rejects_unsupported_answer_and_bad_options(self):
        context = "Naila is a civil engineer. She has five years of experience."
        valid = {
            "question": "What profession does the document identify for Naila?",
            "options": ["A. Civil engineer", "B. Nurse", "C. Chef", "D. Pilot"],
            "answer": "A",
            "explanation": "The document identifies Naila as a civil engineer.",
            "difficulty": "Easy",
        }
        self.assertEqual(len(validate_quiz([valid], context)), 1)

        invalid_answer = dict(valid, answer="B")
        self.assertEqual(validate_quiz([invalid_answer], context), [])

        duplicate_options = dict(valid, options=["A. Engineer", "B. Engineer", "C. Chef", "D. Pilot"])
        self.assertEqual(validate_quiz([duplicate_options], context), [])

        missing_difficulty = dict(valid)
        del missing_difficulty["difficulty"]
        self.assertEqual(validate_quiz([missing_difficulty], context), [])

        three_options = dict(valid, options=["A. Civil engineer", "B. Nurse", "C. Chef"])
        completed = validate_quiz([three_options], context)
        self.assertEqual(len(completed), 1)
        self.assertEqual(len(completed[0]["options"]), 4)
        self.assertIn("Not stated in the document", completed[0]["options"][3])

    def test_generate_quiz_uses_local_assistant_and_validates_output(self):
        payload = [{
            "question": "How many years of experience does she have?",
            "options": ["A. Five years", "B. One year", "C. Ten years", "D. Two years"],
            "answer": "A",
            "explanation": "The document says she has five years of experience.",
            "difficulty": "Medium",
        }]

        class FakeAssistant:
            def generate_content(self, instruction, context, max_new_tokens):
                self.context = context
                self.max_new_tokens = max_new_tokens
                return json.dumps(payload)

        assistant = FakeAssistant()
        questions = generate_quiz("Naila-CV.txt", self.knowledge_base, assistant)
        self.assertEqual(len(questions), 1)
        self.assertEqual(questions[0]["answer"], "A")
        self.assertEqual(questions[0]["difficulty"], "Medium")
        self.assertIn("five years", assistant.context)
        self.assertEqual(assistant.max_new_tokens, 450)

    def test_ingestion_keeps_uploaded_filename_not_temporary_name(self):
        class FakeModel:
            def encode(self, texts, normalize_embeddings=True):
                count = len(texts) if isinstance(texts, list) else 1
                return np.ones((count, 384), dtype="float32")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_file = root / "temporary-upload.txt"
            source_file.write_text(
                "Naila is a civil engineer with five years of experience.",
                encoding="utf-8",
            )
            storage = root / "index"

            with patch("src.knowledge_base.load_embedding_model", return_value=FakeModel()):
                knowledge_base = KnowledgeBase(storage_dir=storage)
                self.assertGreater(
                    knowledge_base.add_document(source_file, source_name="Naila-CV.txt"),
                    0,
                )

            self.assertEqual(knowledge_base.documents_loaded, ["Naila-CV.txt"])
            self.assertEqual(knowledge_base.vector_store.sources, ["Naila-CV.txt"])

    def test_embedding_migration_preserves_chunks_and_source_metadata(self):
        class FakeModel:
            def encode(self, texts, normalize_embeddings=True):
                count = len(texts) if isinstance(texts, list) else 1
                return np.ones((count, 384), dtype="float32")

        with tempfile.TemporaryDirectory() as directory:
            storage = Path(directory) / "index"
            old_store = VectorStore(storage_dir=storage)
            chunks = ["Naila is a civil engineer."]
            sources = ["Naila-CV.txt"]
            old_store.add_documents(
                chunks,
                np.ones((1, 384), dtype="float32"),
                source=sources[0],
            )
            (storage / "documents.json").write_text(
                json.dumps(sources),
                encoding="utf-8",
            )

            with patch("src.knowledge_base.load_embedding_model", return_value=FakeModel()):
                migrated = KnowledgeBase(storage_dir=storage)

            self.assertEqual(migrated.vector_store.text_chunks, chunks)
            self.assertEqual(migrated.vector_store.sources, sources)
            self.assertEqual(migrated.vector_store.index.ntotal, 1)
            self.assertEqual(
                json.loads((storage / "embedding_model.json").read_text(encoding="utf-8"))["model"],
                MODEL_NAME,
            )


if __name__ == "__main__":
    unittest.main()
