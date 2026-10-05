"""
PDF and TXT document processor.
Handles file validation, text extraction, and cleaning.
"""

import re
import unicodedata
from pathlib import Path
from typing import Optional

import pymupdf as fitz

from src.config import MAX_UPLOAD_SIZE_BYTES, ALLOWED_FILE_TYPES


class DocumentProcessingError(Exception):
    """Raised when document processing fails in a recoverable way."""


class ProcessedDocument:
    """Holds extracted and cleaned text along with metadata."""

    def __init__(
        self,
        filename: str,
        text: str,
        page_count: int,
        file_type: str,
        page_texts: Optional[list[str]] = None,
    ):
        self.filename = filename
        self.text = text
        self.page_count = page_count
        self.file_type = file_type
        # page_texts[i] = text for page i+1  (only for PDFs)
        self.page_texts: list[str] = page_texts or []

    def __repr__(self) -> str:
        return (
            f"<ProcessedDocument filename={self.filename!r} "
            f"pages={self.page_count} chars={len(self.text)}>"
        )


# ── Validation ────────────────────────────────────────────────────────────────

def validate_file(filename: str, file_bytes: bytes) -> None:
    """
    Validate file before processing.
    Raises DocumentProcessingError with a user-friendly message on failure.
    """
    suffix = Path(filename).suffix.lstrip(".").lower()

    if suffix not in ALLOWED_FILE_TYPES:
        raise DocumentProcessingError(
            f"Unsupported file type '.{suffix}'. "
            f"Please upload one of: {', '.join(ALLOWED_FILE_TYPES)}."
        )

    if len(file_bytes) == 0:
        raise DocumentProcessingError("The uploaded file appears to be empty.")

    if len(file_bytes) > MAX_UPLOAD_SIZE_BYTES:
        from src.config import MAX_UPLOAD_SIZE_MB
        raise DocumentProcessingError(
            f"File is too large. Maximum allowed size is {MAX_UPLOAD_SIZE_MB} MB."
        )

    # Basic PDF header check
    if suffix == "pdf" and not file_bytes[:5] == b"%PDF-":
        raise DocumentProcessingError(
            "The file does not appear to be a valid PDF."
        )


# ── Text Extraction ───────────────────────────────────────────────────────────

def extract_text_from_pdf(filename: str, file_bytes: bytes) -> ProcessedDocument:
    """Extract text from a PDF using PyMuPDF."""
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as exc:
        raise DocumentProcessingError(
            f"Could not open PDF file. It may be corrupted or encrypted. ({exc})"
        ) from exc

    page_count = len(doc)
    if page_count == 0:
        raise DocumentProcessingError("The PDF has no pages.")

    page_texts: list[str] = []
    for page in doc:
        page_texts.append(page.get_text("text") or "")

    doc.close()

    full_text = "\n\n".join(page_texts)
    cleaned_full = clean_text(full_text)
    cleaned_pages = [clean_text(t) for t in page_texts]

    char_count = len(cleaned_full.strip())
    if char_count < 50:
        raise DocumentProcessingError(
            f"Very little text could be extracted from this PDF ({char_count} characters). "
            "The file may be a scanned image-only PDF. "
            "Try uploading a text-layer PDF or a TXT version of your notes."
        )

    return ProcessedDocument(
        filename=filename,
        text=cleaned_full,
        page_count=page_count,
        file_type="pdf",
        page_texts=cleaned_pages,
    )


def extract_text_from_txt(filename: str, file_bytes: bytes) -> ProcessedDocument:
    """Extract text from a plain-text file."""
    for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            text = file_bytes.decode(encoding)
            break
        except (UnicodeDecodeError, ValueError):
            continue
    else:
        raise DocumentProcessingError(
            "Could not decode the text file. Please save it as UTF-8."
        )

    cleaned = clean_text(text)
    if len(cleaned.strip()) < 50:
        raise DocumentProcessingError(
            "The text file contains very little readable content."
        )

    # Approximate page count for TXT (every ~3000 chars ≈ 1 page)
    approx_pages = max(1, len(cleaned) // 3000)

    return ProcessedDocument(
        filename=filename,
        text=cleaned,
        page_count=approx_pages,
        file_type="txt",
        page_texts=[],
    )


def process_document(filename: str, file_bytes: bytes) -> ProcessedDocument:
    """
    Main entry point. Validate and extract text from a PDF or TXT upload.
    Returns a ProcessedDocument on success.
    Raises DocumentProcessingError with a user-friendly message on failure.
    """
    validate_file(filename, file_bytes)

    suffix = Path(filename).suffix.lstrip(".").lower()
    if suffix == "pdf":
        return extract_text_from_pdf(filename, file_bytes)
    elif suffix == "txt":
        return extract_text_from_txt(filename, file_bytes)
    else:
        raise DocumentProcessingError(f"Unsupported file type: {suffix}")


# ── Text Cleaning ─────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Clean extracted text:
    - Normalize unicode
    - Remove null bytes and control characters (keep newlines/tabs)
    - Collapse excessive blank lines
    - Strip trailing whitespace per line
    """
    if not text:
        return ""

    # Normalize unicode (NFC form)
    text = unicodedata.normalize("NFC", text)

    # Remove null bytes
    text = text.replace("\x00", "")

    # Remove non-printable control chars except \n \r \t
    text = re.sub(r"[^\S\n\r\t ]+", " ", text)

    # Strip each line
    lines = [line.rstrip() for line in text.splitlines()]

    # Collapse multiple blank lines into at most 1 blank line
    cleaned_lines: list[str] = []
    blank_count = 0
    for line in lines:
        if line.strip() == "":
            blank_count += 1
            if blank_count <= 1:
                cleaned_lines.append("")
        else:
            blank_count = 0
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


def sanitize_filename(filename: str) -> str:
    """Return a safe version of a filename (no path traversal, no special chars)."""
    name = Path(filename).name
    name = re.sub(r"[^\w\s\-.]", "_", name)
    name = name.strip(". ")
    return name or "document"
