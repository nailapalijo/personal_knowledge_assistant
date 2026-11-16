
import json
from pathlib import Path

import faiss
import numpy as np


class VectorStore:

    def __init__(self, dimension=384):
        self.dimension = dimension
        self.storage_dir = Path("data/faiss")
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.index_path = self.storage_dir / "index.faiss"
        self.chunks_path = self.storage_dir / "chunks.json"

        self.index = faiss.IndexFlatIP(dimension)
        self.text_chunks = []

        self.load()

    def add_documents(self, chunks, embeddings):

        if not chunks:
            return

        vectors = np.asarray(embeddings, dtype="float32")

        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Expected {self.dimension} dimensions, "
                f"got {vectors.shape[1]}"
            )

        self.index.add(vectors)
        self.text_chunks.extend(chunks)

        self.save()

    def save(self):

        faiss.write_index(self.index, str(self.index_path))

        with open(self.chunks_path, "w", encoding="utf-8") as file:
            json.dump(self.text_chunks, file, ensure_ascii=False)

    def load(self):

        if self.index_path.exists() and self.chunks_path.exists():

            loaded_index = faiss.read_index(str(self.index_path))

            with open(self.chunks_path, "r", encoding="utf-8") as file:
                chunks = json.load(file)

            if loaded_index.d != self.dimension:
                raise ValueError("Saved FAISS index has wrong dimensions.")

            if loaded_index.ntotal != len(chunks):
                raise ValueError("FAISS index and chunks are inconsistent.")

            self.index = loaded_index
            self.text_chunks = chunks

    def search(self, query_embedding, top_k=3):

        if self.index.ntotal == 0:
            return []

        query_vector = np.asarray(
            [query_embedding],
            dtype="float32"
        )

        scores, indices = self.index.search(
            query_vector,
            min(top_k, self.index.ntotal)
        )

        results = []

        for score, idx in zip(scores[0], indices[0]):

            if idx >= 0:
                results.append({
                    "text": self.text_chunks[idx],
                    "score": float(score)
                })

        return results
