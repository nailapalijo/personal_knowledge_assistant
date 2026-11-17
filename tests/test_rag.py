import builtins
from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.rag import NO_ANSWER, NO_ANSWER_URDU, RAGAssistant, format_context


class RagTests(unittest.TestCase):
    def test_local_assistant_does_not_load_models_on_construction(self):
        real_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name == "torch" or name.startswith("transformers"):
                raise AssertionError(f"Unexpected heavy import: {name}")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=guarded_import):
            assistant = RAGAssistant(mode="local")

        self.assertIsNone(assistant.model)
        self.assertIsNone(assistant.tokenizer)

    def test_local_generation_imports_torch_only_when_generating(self):
        class FakeInputs(dict):
            def to(self, device):
                return self

        class FakeOutputs:
            def __getitem__(self, key):
                return [42]

        class FakeTokenizer:
            eos_token_id = 0

            def apply_chat_template(self, messages, **kwargs):
                return "prompt"

            def __call__(self, prompt, **kwargs):
                return FakeInputs(input_ids=SimpleNamespace(shape=(1, 1)))

            def decode(self, tokens, **kwargs):
                return "local answer"

        class FakeModel:
            def generate(self, **kwargs):
                return FakeOutputs()

        assistant = RAGAssistant(mode="local")
        assistant.tokenizer = FakeTokenizer()
        assistant.model = FakeModel()
        assistant.device = "cpu"

        with patch.dict(
            "sys.modules",
            {"torch": SimpleNamespace(inference_mode=nullcontext)}
        ):
            result = assistant._generate([{"role": "user", "content": "Hi"}])

        self.assertEqual(result, "local answer")

    def test_retrieval_records_are_formatted_with_sources(self):
        context = format_context([
            {"source": "notes.txt", "text": "The project began in 2024."},
            {"source": "cv.pdf", "text": "The applicant is an engineer."},
        ])
        self.assertIn("[Source: notes.txt]", context)
        self.assertIn("The project began in 2024.", context)
        self.assertIn("[Source: cv.pdf]", context)

    def test_empty_retrieval_returns_clear_unavailable_answer(self):
        assistant = RAGAssistant.__new__(RAGAssistant)
        self.assertEqual(assistant.generate_answer("Question?", []), NO_ANSWER)
        self.assertEqual(assistant.generate_answer("سوال کیا ہے؟", []), NO_ANSWER_URDU)

    def test_answer_generation_receives_grounded_context(self):
        assistant = RAGAssistant.__new__(RAGAssistant)
        assistant.mode = "local"
        captured = {}

        def fake_generate(messages, max_new_tokens):
            captured["messages"] = messages
            captured["max_new_tokens"] = max_new_tokens
            return "The project began in 2024."

        assistant._generate = fake_generate
        answer = assistant.generate_answer(
            "When did the project begin?",
            [{"source": "notes.txt", "text": "The project began in 2024."}],
        )
        self.assertEqual(answer, "The project began in 2024.")
        prompt = "\n".join(message["content"] for message in captured["messages"])
        self.assertIn("[Source: notes.txt]", prompt)
        self.assertIn("The project began in 2024.", prompt)


if __name__ == "__main__":
    unittest.main()
