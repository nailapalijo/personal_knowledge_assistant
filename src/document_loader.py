from pathlib import Path
from pypdf import PdfReader


MAX_DOCUMENT_SIZE_BYTES = 20 * 1024 * 1024
MAX_PDF_PAGES = 500


def load_pdf(file_path, max_pages=MAX_PDF_PAGES):
    """Extract text from a PDF file."""

    reader = PdfReader(file_path, strict=False)
    if len(reader.pages) > max_pages:
        raise ValueError(
            f"PDF contains more than the {max_pages}-page processing limit."
        )

    text = []
    for page in reader.pages:
        text.append(page.extract_text() or "")

    return "\n".join(text)


def load_txt(file_path):
    """Extract text from a TXT file."""

    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def load_document(file_path):
    """Load PDF or TXT documents."""

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if path.stat().st_size > MAX_DOCUMENT_SIZE_BYTES:
        limit_mb = MAX_DOCUMENT_SIZE_BYTES // (1024 * 1024)
        raise ValueError(
            f"Document exceeds the {limit_mb} MB processing limit."
        )

    if path.suffix.lower() == ".pdf":
        return load_pdf(path)

    elif path.suffix.lower() == ".txt":
        return load_txt(path)

    else:
        raise ValueError("Only PDF and TXT files are supported.")
MAX_DOCUMENT_SIZE_BYTES = 20 * 1024 * 1024
