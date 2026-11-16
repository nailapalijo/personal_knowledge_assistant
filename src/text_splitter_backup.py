
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter


def clean_text(text):
    """Clean extracted PDF/TXT text before chunking."""

    if not text or not isinstance(text, str):
        return ""

    # Remove PDF page-break characters
    text = text.replace("\x0c", "\n")

    # Normalize spaces and tabs
    text = re.sub(r"[ \t]+", " ", text)

    # Keep paragraph boundaries
    text = re.sub(r"\n[ \t]*\n+", "\n\n", text)

    return text.strip()


def split_text(text, chunk_size=700, chunk_overlap=120):
    """Split documents while preserving paragraphs and headings."""

    text = clean_text(text)

    if not text:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            r"\n{2,}",
            r"\n(?=[A-Z][A-Z0-9 &/(),:-]{2,}\s*\n)",
            "\n",
            ". ",
            " ",
            ""
        ],
        is_separator_regex=True,
        keep_separator=True,
        strip_whitespace=True
    )

    chunks = splitter.split_text(text)

    # Remove empty chunks
    return [chunk for chunk in chunks if chunk.strip()]
