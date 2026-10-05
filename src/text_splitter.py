"""
Text chunking module.
Splits cleaned document text into overlapping chunks suitable for embedding.
"""

from dataclasses import dataclass, field
from typing import Optional

from src.config import CHUNK_SIZE, CHUNK_OVERLAP


@dataclass
class TextChunk:
    """A single text chunk with metadata."""
    text: str
    chunk_index: int
    source_filename: str
    page_number: Optional[int] = None  # None for TXT files
    char_start: int = 0
    char_end: int = 0


def split_text_into_chunks(
    text: str,
    source_filename: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[TextChunk]:
    """
    Split text into overlapping chunks.
    Uses paragraph-aware splitting to avoid breaking sentences mid-thought.
    """
    if not text or not text.strip():
        return []

    # First split by paragraphs
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks: list[TextChunk] = []
    current_chunk_parts: list[str] = []
    current_length = 0
    chunk_index = 0

    for para in paragraphs:
        para_len = len(para)

        # If a single paragraph is longer than chunk_size, split it by sentences
        if para_len > chunk_size:
            sub_parts = _split_by_sentences(para, chunk_size)
        else:
            sub_parts = [para]

        for part in sub_parts:
            part_len = len(part)

            if current_length + part_len + 1 > chunk_size and current_chunk_parts:
                # Emit the current chunk
                chunk_text = "\n\n".join(current_chunk_parts)
                chunks.append(
                    TextChunk(
                        text=chunk_text,
                        chunk_index=chunk_index,
                        source_filename=source_filename,
                        char_start=0,
                        char_end=len(chunk_text),
                    )
                )
                chunk_index += 1

                # Overlap: keep the last few parts
                overlap_parts = _compute_overlap(current_chunk_parts, chunk_overlap)
                current_chunk_parts = overlap_parts
                current_length = sum(len(p) for p in overlap_parts)

            current_chunk_parts.append(part)
            current_length += part_len + 1  # +1 for separator

    # Emit the remaining chunk
    if current_chunk_parts:
        chunk_text = "\n\n".join(current_chunk_parts)
        chunks.append(
            TextChunk(
                text=chunk_text,
                chunk_index=chunk_index,
                source_filename=source_filename,
                char_start=0,
                char_end=len(chunk_text),
            )
        )

    return chunks


def split_pages_into_chunks(
    page_texts: list[str],
    source_filename: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[TextChunk]:
    """
    Chunk a list of per-page texts, preserving page number metadata.
    """
    all_chunks: list[TextChunk] = []
    global_index = 0

    for page_num, page_text in enumerate(page_texts, start=1):
        if not page_text.strip():
            continue

        page_chunks = split_text_into_chunks(
            page_text,
            source_filename=source_filename,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        for chunk in page_chunks:
            chunk.chunk_index = global_index
            chunk.page_number = page_num
            all_chunks.append(chunk)
            global_index += 1

    return all_chunks


# ── Helpers ───────────────────────────────────────────────────────────────────

def _split_by_sentences(text: str, max_length: int) -> list[str]:
    """Naively split a long string at sentence boundaries."""
    import re
    sentences = re.split(r"(?<=[.!?])\s+", text)
    parts: list[str] = []
    current = ""
    for sentence in sentences:
        if len(current) + len(sentence) + 1 > max_length and current:
            parts.append(current.strip())
            current = sentence
        else:
            current = (current + " " + sentence).strip() if current else sentence
    if current:
        parts.append(current.strip())
    return parts


def _compute_overlap(parts: list[str], overlap_chars: int) -> list[str]:
    """Return the trailing parts of a chunk list that together reach ~overlap_chars."""
    overlap: list[str] = []
    total = 0
    for part in reversed(parts):
        overlap.insert(0, part)
        total += len(part)
        if total >= overlap_chars:
            break
    return overlap
