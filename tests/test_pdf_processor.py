"""
Tests for PDF and document processing.
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.pdf_processor import (
    clean_text,
    sanitize_filename,
    validate_file,
    DocumentProcessingError,
    extract_text_from_txt,
)


# ── clean_text ─────────────────────────────────────────────────────────────────

def test_clean_text_basic():
    raw = "  Hello   World  \n\n\n\nFoo  "
    result = clean_text(raw)
    assert "Hello" in result
    assert "World" in result
    assert "Foo" in result


def test_clean_text_null_bytes():
    raw = "Hello\x00World"
    result = clean_text(raw)
    assert "\x00" not in result
    assert "Hello" in result
    assert "World" in result


def test_clean_text_empty():
    assert clean_text("") == ""
    assert clean_text("   ") == ""


def test_clean_text_many_blank_lines():
    raw = "Line1\n\n\n\n\n\nLine2"
    result = clean_text(raw)
    # Should have at most 2 consecutive blank lines
    assert "\n\n\n" not in result


# ── sanitize_filename ─────────────────────────────────────────────────────────

def test_sanitize_filename_normal():
    assert sanitize_filename("my notes.pdf") == "my notes.pdf"


def test_sanitize_filename_path_traversal():
    result = sanitize_filename("../../../etc/passwd")
    assert ".." not in result
    assert "/" not in result


def test_sanitize_filename_special_chars():
    result = sanitize_filename("notes; rm -rf *.pdf")
    assert ";" not in result


# ── validate_file ─────────────────────────────────────────────────────────────

def test_validate_unsupported_type():
    with pytest.raises(DocumentProcessingError, match="Unsupported file type"):
        validate_file("notes.docx", b"some content")


def test_validate_empty_file():
    with pytest.raises(DocumentProcessingError, match="empty"):
        validate_file("notes.pdf", b"")


def test_validate_fake_pdf():
    with pytest.raises(DocumentProcessingError, match="valid PDF"):
        validate_file("notes.pdf", b"This is not a PDF")


def test_validate_valid_txt():
    # Should not raise
    validate_file("notes.txt", b"Hello World " * 10)


# ── TXT extraction ────────────────────────────────────────────────────────────

def test_extract_txt_utf8():
    content = b"This is a test document with enough content to pass validation. " * 5
    result = extract_text_from_txt("test.txt", content)
    assert "test document" in result.text
    assert result.file_type == "txt"
    assert result.page_count >= 1


def test_extract_txt_too_short():
    with pytest.raises(DocumentProcessingError, match="very little"):
        extract_text_from_txt("test.txt", b"Hi")


def test_extract_txt_latin1():
    content = "Héllo Wörld - test document with enough content to validate. " * 5
    result = extract_text_from_txt("test.txt", content.encode("latin-1"))
    assert result.text  # Should decode without error


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
