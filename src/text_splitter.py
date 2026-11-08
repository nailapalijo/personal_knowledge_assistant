
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_text(text, chunk_size=800, chunk_overlap=150):
    """Split documents into meaningful overlapping chunks."""

    if not text or not text.strip():
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    return splitter.split_text(text)
