
import json
import re
from pathlib import Path

from rank_bm25 import BM25Plus

from src.embeddings import (
    MODEL_NAME,
    create_embeddings,
    create_query_embedding,
    load_embedding_model,
)
from src.document_loader import load_document
from src.text_splitter import split_text
from src.vector_store import VectorStore


STOP_WORDS = {
    "what", "is", "are", "am", "my", "your", "the",
    "a", "an", "of", "in", "on", "to", "for", "and",
    "me", "i", "you", "who", "how", "where", "when",
    "does", "do", "can", "could", "tell", "about",
    "please", "give", "show", "explain", "this", "that",
    "it", "with", "from", "was", "were", "be", "has",
    "have", "had", "which", "why", "describe", "define",
    "document", "information", "find", "answer"
}

URDU_STOP_WORDS = {
    "کیا", "ہے", "ہیں", "میں", "کا", "کی", "کے", "کو", "سے",
    "اور", "پر", "یہ", "وہ", "آپ", "میرا", "میری", "میرے",
    "بتائیں", "براہ", "کرم"
}


def tokenize(text):
    if not text:
        return []

    tokens = re.findall(r"[^\W_]+(?:[+#.][^\W_]+)*", text.casefold(), re.UNICODE)
    normalized = []

    for word in tokens:
        if word in STOP_WORDS or word in URDU_STOP_WORDS:
            continue

        if word.isascii() and word.endswith("ies") and len(word) > 4:
            word = word[:-3] + "y"
        elif (
            word.isascii()
            and word.endswith("s")
            and len(word) > 4
            and not word.endswith("ss")
        ):
            word = word[:-1]

        normalized.append(word)

    return normalized


class KnowledgeBase:

    def __init__(self, storage_dir="data/faiss"):
        print("Loading embedding model...")

        self.model = load_embedding_model()
        self.storage_dir = Path(storage_dir)
        self.vector_store = VectorStore(storage_dir=self.storage_dir)

        self.metadata_path = self.storage_dir / "documents.json"
        self.chunks_path = self.storage_dir / "chunks.json"
        self.embedding_metadata_path = self.storage_dir / "embedding_model.json"

        self.metadata_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        if self.metadata_path.exists():
            with open(self.metadata_path, "r", encoding="utf-8") as file:
                self.documents_loaded = json.load(file)
        else:
            self.documents_loaded = []

        self._ensure_current_embeddings()

        self.bm25 = None
        self.bm25_chunks = []
        self.bm25_sources = []

        self._build_bm25()

        print("Knowledge Base initialized successfully!")

    def _ensure_current_embeddings(self):
        chunks, sources = self._read_chunks()

        saved_model = None
        if self.embedding_metadata_path.exists():
            with open(self.embedding_metadata_path, "r", encoding="utf-8") as file:
                saved_model = json.load(file).get("model")

        if chunks and saved_model != MODEL_NAME:
            embeddings = create_embeddings(
                chunks,
                self.model
            )
            self.vector_store.replace_documents(chunks, embeddings, sources)

        with open(self.embedding_metadata_path, "w", encoding="utf-8") as file:
            json.dump({"model": MODEL_NAME}, file, ensure_ascii=False, indent=2)

    def _read_chunks(self):

        if not self.chunks_path.exists():
            return [], []

        with open(self.chunks_path, "r", encoding="utf-8") as file:
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

        chunks, sources = self._read_chunks()

        self.bm25 = None
        self.bm25_chunks = chunks
        self.bm25_sources = sources

        if not chunks:
            return

        tokenized_chunks = [tokenize(chunk) for chunk in chunks]

        if not any(tokenized_chunks):
            return

        self.bm25 = BM25Plus(tokenized_chunks)

    def add_document(self, file_path, source_name=None):

        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Document not found: {path}"
            )

        source_name = Path(source_name or path.name).name

        if source_name in self.documents_loaded:
            print(f"{source_name} is already processed.")
            return 0

        text = load_document(path)

        if not text or not text.strip():
            print("Document contains no extractable text.")
            return 0

        chunks = split_text(text)

        if not chunks:
            print("No valid text chunks were created.")
            return 0

        embeddings = create_embeddings(
            chunks,
            self.model
        )

        self.vector_store.add_documents(
            chunks,
            embeddings,
            source=source_name
        )

        self.documents_loaded.append(source_name)

        with open(self.metadata_path, "w", encoding="utf-8") as file:
            json.dump(
                self.documents_loaded,
                file,
                ensure_ascii=False,
                indent=4
            )

        self._build_bm25()

        print(
            f"Successfully processed {source_name}: "
            f"{len(chunks)} chunks"
        )

        return len(chunks)

    def search_documents(self, question, top_k=3):

        if not question or not question.strip():
            return []

        if self.vector_store.index.ntotal == 0:
            return []

        query_tokens = tokenize(question)
        query_terms = set(query_tokens)

        query_embedding = create_query_embedding(question, self.model)

        candidate_count = min(
            max(top_k * 5, 10),
            self.vector_store.index.ntotal
        )

        semantic_results = self.vector_store.search(
            query_embedding,
            top_k=candidate_count
        )

        keyword_results = []

        if self.bm25 is not None and query_tokens:

            bm25_scores = self.bm25.get_scores(query_tokens)

            ranked_indices = sorted(
                range(len(bm25_scores)),
                key=lambda i: bm25_scores[i],
                reverse=True
            )

            for index in ranked_indices[:candidate_count]:

                keyword_results.append({
                    "text": self.bm25_chunks[index],
                    "source": self.bm25_sources[index],
                    "bm25_score": float(bm25_scores[index])
                })

        max_bm25 = max(
            (item["bm25_score"] for item in keyword_results),
            default=0.0
        )

        combined = {}

        for item in semantic_results:

            key = (item["source"], item["text"])

            combined[key] = {
                "text": item["text"],
                "source": item["source"],
                "semantic_score": float(item["score"]),
                "keyword_score": 0.0
            }

        for item in keyword_results:

            key = (item["source"], item["text"])

            normalized_bm25 = (
                item["bm25_score"] / max_bm25
                if max_bm25 > 0
                else 0.0
            )

            if key not in combined:

                combined[key] = {
                    "text": item["text"],
                    "source": item["source"],
                    "semantic_score": 0.0,
                    "keyword_score": normalized_bm25
                }

            else:

                combined[key]["keyword_score"] = normalized_bm25

        results = []

        for item in combined.values():

            chunk_tokens = set(tokenize(item["text"]))

            overlap_ratio = (
                len(query_terms.intersection(chunk_tokens))
                / len(query_terms)
                if query_terms
                else 0.0
            )

            semantic_score = item["semantic_score"]
            keyword_score = item["keyword_score"]

            # Initial relevance filter.
            # These thresholds will be tuned using real document tests.
            is_relevant = (
                semantic_score >= 0.28
                or overlap_ratio >= 0.25
            )

            if not is_relevant:
                continue

            final_score = (
                semantic_score * 0.75
                + keyword_score * 0.15
                + overlap_ratio * 0.10
            )

            results.append({
                "text": item["text"],
                "source": item["source"],
                "score": float(final_score),
                "semantic_score": float(semantic_score),
                "keyword_score": float(keyword_score),
                "overlap_ratio": float(overlap_ratio)
            })

        results.sort(
            key=lambda item: item["score"],
            reverse=True
        )

        return results[:top_k]
