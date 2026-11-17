import builtins
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.knowledge_base import KnowledgeBase
from src.rag import NO_ANSWER, RAGAssistant
from src.runtime import get_runtime_mode


class CloudModeTests(unittest.TestCase):
    def test_runtime_defaults_are_safe_for_cloud_and_local_windows(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(get_runtime_mode(platform="linux"), "cloud")
            self.assertEqual(get_runtime_mode(platform="nt"), "local")

    def test_cloud_assistant_does_not_import_or_load_generator(self):
        real_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name == "torch" or name.startswith("transformers"):
                raise AssertionError(f"Unexpected heavy import: {name}")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=guarded_import):
            assistant = RAGAssistant(mode="cloud")
            answer = assistant.generate_answer(
                "When did the project start?",
                [{
                    "source": "notes.txt",
                    "text": "The project started in 2024."
                }],
            )

        self.assertFalse(assistant.supports_generation)
        self.assertIsNone(assistant.model)
        self.assertIn("2024", answer)
        self.assertIn("notes.txt", answer)
        self.assertEqual(assistant.generate_answer("unrelated", []), NO_ANSWER)

    def test_cloud_knowledge_base_uses_local_hashing_without_model_download(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            document = root / "notes.txt"
            document.write_text(
                "The project started in 2024 and serves local users.",
                encoding="utf-8",
            )
            index_dir = root / "index"

            with patch(
                "src.knowledge_base.load_embedding_model",
                side_effect=AssertionError("Local model must not load in Cloud mode"),
            ):
                knowledge_base = KnowledgeBase(
                    storage_dir=index_dir,
                    mode="cloud",
                )
                self.assertIsNone(knowledge_base.model)
                self.assertGreater(
                    knowledge_base.add_document(document, source_name="notes.txt"),
                    0,
                )
                results = knowledge_base.search_documents(
                    "When did the project start?",
                    top_k=3,
                )

            self.assertIsNotNone(knowledge_base.model)
            self.assertTrue(results)
            self.assertEqual(results[0]["source"], "notes.txt")


if __name__ == "__main__":
    unittest.main()
