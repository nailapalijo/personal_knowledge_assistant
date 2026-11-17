import unittest

from src.embeddings import (
    HashingEmbeddingModel,
    create_embeddings,
    create_query_embedding,
)


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

    def test_hashing_embeddings_are_normalized_and_dimensioned(self):
        model = HashingEmbeddingModel()
        vectors = create_embeddings(["project started in 2024"], model)
        query = create_query_embedding("project started in 2024", model)

        self.assertEqual(vectors.shape, (1, 384))
        self.assertAlmostEqual(float((vectors[0] @ query)), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
