import unittest

from src.rag import NO_ANSWER, NO_ANSWER_URDU, RAGAssistant, format_context


class RagTests(unittest.TestCase):
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
