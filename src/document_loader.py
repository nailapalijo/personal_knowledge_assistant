from pathlib import Path
from pypdf import PdfReader


def load_pdf(file_path):
    """Extract text from a PDF file."""

    reader = PdfReader(file_path)
    text = ""

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        text += page_text + "\n"

    return text


def load_txt(file_path):
    """Extract text from a TXT file."""

    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def load_document(file_path):
    """Load PDF or TXT documents."""

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if path.suffix.lower() == ".pdf":
        return load_pdf(path)

    elif path.suffix.lower() == ".txt":
        return load_txt(path)

    else:
        raise ValueError("Only PDF and TXT files are supported.")