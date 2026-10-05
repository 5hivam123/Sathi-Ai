"""
Tests for text chunking.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src.text_splitter import (
    split_text_into_chunks,
    split_pages_into_chunks,
    TextChunk,
)

SAMPLE_TEXT = """
Introduction to Operating Systems

An operating system (OS) is system software that manages computer hardware and software resources.
It provides common services for computer programs.

Process Management

A process is an instance of a program that is being executed.
The operating system manages processes through scheduling algorithms.

Types of Scheduling

First Come First Served (FCFS) is the simplest scheduling algorithm.
Shortest Job First (SJF) selects the process with the smallest execution time.
Round Robin (RR) gives each process a fixed time slice in a cyclic way.

Memory Management

Memory management is the process of controlling and coordinating computer memory.
The OS handles allocation and deallocation of memory blocks.

Virtual Memory

Virtual memory is a memory management technique that allows execution of processes that
may not be completely in main memory. It gives the programmer the illusion of a large main memory.

File Systems

A file system controls how data is stored and retrieved on a storage device.
Common file systems include FAT32, NTFS, ext4, and APFS.

Security and Protection

Operating systems must protect themselves and user programs from malicious or erroneous code.
Protection mechanisms include user authentication, access control, and encryption.
""".strip()


def test_basic_chunking():
    chunks = split_text_into_chunks(SAMPLE_TEXT, source_filename="test.pdf")
    assert len(chunks) > 0
    for chunk in chunks:
        assert isinstance(chunk, TextChunk)
        assert len(chunk.text) > 0
        assert chunk.source_filename == "test.pdf"


def test_chunk_size_respected():
    chunks = split_text_into_chunks(SAMPLE_TEXT, source_filename="test.pdf", chunk_size=300)
    for chunk in chunks:
        # Allow slight overshoot at sentence boundaries
        assert len(chunk.text) <= 500, f"Chunk too large: {len(chunk.text)}"


def test_empty_text_returns_no_chunks():
    chunks = split_text_into_chunks("", source_filename="test.pdf")
    assert chunks == []


def test_whitespace_only_returns_no_chunks():
    chunks = split_text_into_chunks("   \n\n\n   ", source_filename="test.pdf")
    assert chunks == []


def test_chunk_indices_are_sequential():
    chunks = split_text_into_chunks(SAMPLE_TEXT, source_filename="test.pdf")
    for i, chunk in enumerate(chunks):
        assert chunk.chunk_index == i


def test_page_chunking_preserves_page_numbers():
    pages = [
        "Page 1 content about operating systems. " * 20,
        "Page 2 content about process management. " * 20,
        "Page 3 content about memory management. " * 20,
    ]
    chunks = split_pages_into_chunks(pages, source_filename="test.pdf")
    assert len(chunks) > 0
    page_numbers = {chunk.page_number for chunk in chunks}
    assert 1 in page_numbers
    assert 2 in page_numbers
    assert 3 in page_numbers


def test_page_chunking_skips_empty_pages():
    pages = [
        "Content on page 1. " * 20,
        "",  # empty page
        "Content on page 3. " * 20,
    ]
    chunks = split_pages_into_chunks(pages, source_filename="test.pdf")
    page_numbers = {chunk.page_number for chunk in chunks}
    assert 2 not in page_numbers  # empty page should be skipped


def test_short_text_produces_one_chunk():
    text = "Short text."
    chunks = split_text_into_chunks(text, source_filename="test.pdf")
    assert len(chunks) == 1
    assert "Short text" in chunks[0].text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
