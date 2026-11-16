import unittest

from src.embeddings import create_embeddings, create_query_embedding


class EmbeddingTests(unittest.TestCase):
    def test_e5_uses_passage_and_query_prefixes(self):
        class FakeModel:
            def encode(self, texts, normalize_embeddings):
                self.texts = texts
                self.normalize_embeddings = normalize_embeddings
                return texts

        model = FakeModel()
        self.assertEqual(create_embeddings(["a document"], model), ["passage: a document"])
        self.assertEqual(model.texts, ["passage: a document"])
        self.assertTrue(model.normalize_embeddings)

        self.assertEqual(create_query_embedding("a question", model), "query: a question")
        self.assertEqual(model.texts, "query: a question")


if __name__ == "__main__":
    unittest.main()
