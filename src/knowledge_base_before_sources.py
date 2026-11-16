
import json
from pathlib import Path

from src.document_loader import load_document
from src.text_splitter import split_text
from src.embeddings import load_embedding_model
from src.vector_store import VectorStore


class KnowledgeBase:

    def __init__(self):

        print("Loading embedding model...")

        self.model = load_embedding_model()
        self.vector_store = VectorStore()

        self.metadata_path = Path("data/faiss/documents.json")
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)

        if self.metadata_path.exists():
            with open(self.metadata_path, "r", encoding="utf-8") as file:
                self.documents_loaded = json.load(file)
        else:
            self.documents_loaded = []

        print("Knowledge Base initialized successfully!")

    def add_document(self, file_path):

        path = Path(file_path)

        if path.name in self.documents_loaded:
            print(f"{path.name} is already processed.")
            return 0

        text = load_document(path)

        if not text.strip():
            return 0

        chunks = split_text(text)

        if not chunks:
            return 0

        embeddings = self.model.encode(
            chunks,
            normalize_embeddings=True
        )

        self.vector_store.add_documents(chunks, embeddings)

        self.documents_loaded.append(path.name)

        with open(self.metadata_path, "w", encoding="utf-8") as file:
            json.dump(
                self.documents_loaded,
                file,
                ensure_ascii=False,
                indent=4
            )

        return len(chunks)
import json
from pathlib import Path

from src.document_loader import load_document
from src.text_splitter import split_text
from src.embeddings import load_embedding_model
from src.vector_store import VectorStore


class KnowledgeBase:

    def __init__(self):

        print("Loading embedding model...")

        self.model = load_embedding_model()
        self.vector_store = VectorStore()

        self.metadata_path = Path("data/faiss/documents.json")
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)

        if self.metadata_path.exists():
            with open(self.metadata_path, "r", encoding="utf-8") as file:
                self.documents_loaded = json.load(file)
        else:
            self.documents_loaded = []

        print("Knowledge Base initialized successfully!")

    def add_document(self, file_path):

        path = Path(file_path)

        if path.name in self.documents_loaded:
            print(f"{path.name} is already processed.")
            return 0

        text = load_document(path)

        if not text.strip():
            return 0

        chunks = split_text(text)

        if not chunks:
            return 0

        embeddings = self.model.encode(
            chunks,
            normalize_embeddings=True
        )

        self.vector_store.add_documents(
            chunks,
            embeddings,
            source=path.name
        )

        self.documents_loaded.append(path.name)

        with open(self.metadata_path, "w", encoding="utf-8") as file:
            json.dump(
                self.documents_loaded,
                file,
                ensure_ascii=False,
                indent=4
            )

        return len(chunks)

    def search_documents(self, question, top_k=3):

        query_embedding = self.model.encode(
            question,
            normalize_embeddings=True
        )

        return self.vector_store.search(
            query_embedding,
            top_k=top_k
        )


    def search_documents(self, question, top_k=3):

        query_embedding = self.model.encode(
            question,
            normalize_embeddings=True
        )

        return self.vector_store.search(
            query_embedding,
            top_k=top_k
        )
