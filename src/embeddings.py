
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_embedding_model():
    """Load the Hugging Face embedding model."""
    return SentenceTransformer(MODEL_NAME)


def create_embeddings(texts, model):
    """Convert text into numerical embeddings."""
    if not texts:
        return []

    embeddings = model.encode(
        texts,
        normalize_embeddings=True
    )

    return embeddings
