
from pathlib import Path

from src.document_loader import load_document
from src.text_splitter import split_text
from src.embeddings import load_embedding_model
from src.vector_store import VectorStore


class KnowledgeBase:

    def __init__(self):
        self.model = load_embedding_model()
        self.vector_store = VectorStore()
        self.documents_loaded = []

    def add_document(self, file_path):
        """Load, split and store a document."""

        path = Path(file_path)

        text = load_document(path)

        if not text.strip():
            return 0

        chunks = split_text(text)

        embeddings = self.model.encode(
            chunks,
            normalize_embeddings=True
        )

        self.vector_store.add_documents(
            chunks,
            embeddings
        )

        self.documents_loaded.append(path.name)

        return len(chunks)

    def search_documents(self, question, top_k=3):
        """Search for relevant document information."""

        query_embedding = self.model.encode(
            question,
            normalize_embeddings=True
        )

        return self.vector_store.search(
            query_embedding,
            top_k=top_k
        )
