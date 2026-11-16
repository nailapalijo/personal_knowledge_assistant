from src.knowledge_base import KnowledgeBase
kb = KnowledgeBase()
questions = [
    "What is my full name?",
    "What is my educational background?",
    "What are my technical skills?"
]
for question in questions:
    print("\nQUESTION:", question)
    embedding = kb.model.encode(question, normalize_embeddings=True)
    results = kb.vector_store.search(embedding, top_k=10)
    for i, item in enumerate(results, 1):
        print(f"\nRank: {i} | Score: {item['score']:.4f}")
        print("Source:", item["source"])
        print("Text:", item["text"][:180])
