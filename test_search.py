
from src.knowledge_base import KnowledgeBase

print("\nLoading Knowledge Base...")

kb = KnowledgeBase()

questions = [
    "What is my full name?",
    "What is my educational background?",
    "What are my technical skills?",
    "What is my work experience?"
]

for question in questions:

    print("\n" + "=" * 60)
    print("QUESTION:", question)
    print("=" * 60)

    results = kb.search_documents(question, top_k=3)

    for result in results:

        print("\nSimilarity Score:", round(result["score"], 4))
        print("Source:", result["source"])
        print("Text:", result["text"][:300])
