
import json
import re
from pathlib import Path

from rank_bm25 import BM25Plus

from src.document_loader import load_document
from src.text_splitter import split_text
from src.embeddings import load_embedding_model
from src.vector_store import VectorStore


STOP_WORDS = {
    "what", "is", "are", "am", "my", "your", "the",
    "a", "an", "of", "in", "on", "to", "for", "and",
    "me", "i", "you", "who", "how", "where", "when",
    "does", "do", "can", "could", "tell", "about",
    "please", "give", "show", "explain"
}


def tokenize(text):
    """Normalize text into searchable keywords."""
    if not text:
        return []

    tokens = re.findall(
        r"[a-zA-Z0-9+#.]+",
        text.lower()
    )

    normalized = []

    for word in tokens:
        if word in STOP_WORDS:
            continue

        # Simple normalization for common plurals
        if word.endswith("ies") and len(word) > 4:
            word = word[:-3] + "y"
        elif word.endswith("s") and len(word) > 4 and not word.endswith("ss"):
            word = word[:-1]

        normalized.append(word)

    return normalized


class KnowledgeBase:

    def __init__(self):
        print("Loading embedding model...")

        self.model = load_embedding_model()
        self.vector_store = VectorStore()

        self.metadata_path = Path("data/faiss/documents.json")
        self.chunks_path = Path("data/faiss/chunks.json")

        self.metadata_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        if self.metadata_path.exists():
            with open(
                self.metadata_path,
                "r",
                encoding="utf-8"
            ) as file:
                self.documents_loaded = json.load(file)
        else:
            self.documents_loaded = []

        self.bm25 = None
        self.bm25_chunks = []
        self.bm25_sources = []

        self._build_bm25()

        print("Knowledge Base initialized successfully!")

    def _read_chunks(self):
        """Read stored chunks and their source names."""

        if not self.chunks_path.exists():
            return [], []

        with open(
            self.chunks_path,
            "r",
            encoding="utf-8"
        ) as file:
            data = json.load(file)

        if isinstance(data, list):
            chunks = data
            sources = ["Unknown document"] * len(chunks)
        else:
            chunks = data.get("chunks", [])
            sources = data.get("sources", [])

        if len(sources) != len(chunks):
            sources = ["Unknown document"] * len(chunks)

        return chunks, sources

    def _build_bm25(self):
        """Build keyword search index from stored chunks."""

        chunks, sources = self._read_chunks()

        self.bm25 = None
        self.bm25_chunks = chunks
        self.bm25_sources = sources

        if not chunks:
            return

        tokenized_chunks = [
            tokenize(chunk) for chunk in chunks
        ]

        if not any(tokenized_chunks):
            return

        self.bm25 = BM25Plus(tokenized_chunks)

    def add_document(self, file_path):
        """Load, split, embed and store a new document."""

        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Document not found: {path}"
            )

        if path.name in self.documents_loaded:
            print(f"{path.name} is already processed.")
            return 0

        text = load_document(path)

        if not text or not text.strip():
            print("Document contains no extractable text.")
            return 0

        chunks = split_text(text)

        if not chunks:
            print("No valid text chunks were created.")
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

        with open(
            self.metadata_path,
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(
                self.documents_loaded,
                file,
                ensure_ascii=False,
                indent=4
            )

        self._build_bm25()

        print(
            f"Successfully processed {path.name}: "
            f"{len(chunks)} chunks"
        )

        return len(chunks)

    def search_documents(self, question, top_k=3):
        """Combine semantic and keyword retrieval using RRF."""

        if not question or not question.strip():
            return []

        if self.vector_store.index.ntotal == 0:
            return []

        query_embedding = self.model.encode(
            question,
            normalize_embeddings=True
        )

        # Semantic retrieval
        semantic_results = self.vector_store.search(
            query_embedding,
            top_k=self.vector_store.index.ntotal
        )

        # Keyword retrieval
        keyword_results = []

        if self.bm25 is not None:
            query_tokens = tokenize(question)

            if query_tokens:
                scores = self.bm25.get_scores(query_tokens)

                ranked_indices = sorted(
                    range(len(scores)),
                    key=lambda index: scores[index],
                    reverse=True
                )

                for index in ranked_indices:
                    keyword_results.append({
                        "text": self.bm25_chunks[index],
                        "source": self.bm25_sources[index],
                        "score": float(scores[index])
                    })

        # Reciprocal Rank Fusion
        combined = {}
        rrf_constant = 60

        for result_list in (
            semantic_results,
            keyword_results
        ):
            seen = set()

            for rank, item in enumerate(result_list, start=1):
                key = (
                    item["source"],
                    item["text"]
                )

                if key in seen:
                    continue

                seen.add(key)

                combined[key] = (
                    combined.get(key, 0)
                    + 1 / (rrf_constant + rank)
                )

        ranked_results = sorted(
            combined.items(),
            key=lambda item: item[1],
            reverse=True
        )

        results = []

        for (source, chunk), score in ranked_results[:top_k]:
            results.append({
                "text": chunk,
                "source": source,
                "score": score
            })

        return results
