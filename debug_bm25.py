import json
from rank_bm25 import BM25Okapi
from src.knowledge_base import tokenize

with open("data/faiss/chunks.json", "r", encoding="utf-8") as file:
    data = json.load(file)

chunks = data["chunks"]
sources = data["sources"]

tokenized_chunks = [tokenize(chunk) for chunk in chunks]
bm25 = BM25Okapi(tokenized_chunks)

question = "What is my full name?"
query_tokens = tokenize(question)

scores = bm25.get_scores(query_tokens)

for i, score in enumerate(scores):
    print(f"\nChunk {i + 1} | Score: {score:.4f}")
    print(f"Source: {sources[i]}")
    print(chunks[i][:200])
