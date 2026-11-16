
from sentence_transformers import SentenceTransformer


MODEL_NAME = "intfloat/multilingual-e5-small"


def load_embedding_model():
    """Load the multilingual Hugging Face embedding model for local inference."""
    return SentenceTransformer(MODEL_NAME)


def create_embeddings(texts, model):
    """Convert text into numerical embeddings."""
    if not texts:
        return []

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
