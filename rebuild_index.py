
import json
import faiss
from pathlib import Path

from src.knowledge_base import KnowledgeBase


def main():

    print("\nStarting document re-indexing...\n")

    kb = KnowledgeBase()

    upload_folder = Path("data/uploads")

    files = sorted(
        file for file in upload_folder.iterdir()
        if file.is_file() and file.suffix.lower() in [".pdf", ".txt"]
    )

    if not files:
        print("No PDF or TXT documents found.")
        return

    # Create a fresh FAISS index
    kb.vector_store.index = faiss.IndexFlatIP(
        kb.vector_store.dimension
    )

    kb.vector_store.text_chunks = []
    kb.vector_store.sources = []
    kb.documents_loaded = []

    # Reset saved document metadata
    with open(kb.metadata_path, "w", encoding="utf-8") as file:
        json.dump([], file)

    # Process every uploaded document again
    for path in files:

        try:
            print(f"\nProcessing: {path.name}")

            chunks = kb.add_document(path)

            print(f"Chunks created: {chunks}")

        except Exception as error:
            print(f"Error processing {path.name}: {error}")

    print("\n========== RE-INDEXING SUMMARY ==========")
    print("Documents processed:", len(kb.documents_loaded))
    print("Total chunks:", kb.vector_store.index.ntotal)
    print("Sources saved:", len(kb.vector_store.sources))

    print("\nSource distribution:")

    for source in sorted(set(kb.vector_store.sources)):
        print(
            source,
            "->",
            kb.vector_store.sources.count(source),
            "chunks"
        )

    print("\nRe-indexing completed!")


if __name__ == "__main__":
    main()
