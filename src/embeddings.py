
import hashlib
import re

import numpy as np


MODEL_NAME = "intfloat/multilingual-e5-small"
HASHING_MODEL_NAME = "hashing-multilingual-384-v1"
EMBEDDING_DIMENSION = 384


def load_embedding_model():
    """Load the multilingual Hugging Face embedding model for local inference."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_NAME)


class HashingEmbeddingModel:
    """Small deterministic multilingual lexical vectors for Cloud retrieval."""

    def encode(self, texts, normalize_embeddings=True):
        if isinstance(texts, str):
            return self._encode_one(texts)

        vectors = np.vstack([self._encode_one(text) for text in texts])
        if normalize_embeddings:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            np.divide(vectors, norms, out=vectors, where=norms != 0)
        return vectors

    @staticmethod
    def _encode_one(text):
        vector = np.zeros(EMBEDDING_DIMENSION, dtype="float32")
        words = re.findall(r"[^\W_]+", text.casefold(), re.UNICODE)
        features = []

        for word in words:
            if word in {"passage", "query"}:
                continue
            features.append((word, 1.0))
            padded = f"^{word}$"
            features.extend(
                (padded[index:index + 3], 0.25)
                for index in range(max(len(padded) - 2, 0))
            )

        for feature, weight in features:
            digest = hashlib.blake2b(
                feature.encode("utf-8"),
                digest_size=8
            ).digest()
            value = int.from_bytes(digest[:4], "little")
            bucket = value % EMBEDDING_DIMENSION
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[bucket] += sign * weight

        norm = np.linalg.norm(vector)
        if norm:
            vector /= norm
        return vector


def load_cloud_embedding_model():
    return HashingEmbeddingModel()


def create_embeddings(texts, model):
    """Convert text into numerical embeddings."""
    if not texts:
        return np.empty((0, EMBEDDING_DIMENSION), dtype="float32")

    embeddings = model.encode(
        [f"passage: {text}" for text in texts],
        normalize_embeddings=True
    )

    return embeddings


def create_query_embedding(query, model):
    """Encode a retrieval query using the E5 query prefix."""
    return model.encode(
        f"query: {query}",
        normalize_embeddings=True
    )
