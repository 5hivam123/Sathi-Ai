"""
RAG (Retrieval-Augmented Generation) pipeline.
Ties together embeddings, vector search, and LLM generation.
Includes an intelligent direct extraction fallback when local LLM server is offline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from src.vector_store import VectorStore, VectorStoreError
from src.embeddings import embed_query
from src.gemma import generate, LLMError
from src.prompts import (
    build_context_block,
    STUDY_ASSISTANT_PROMPT,
)
from src.config import TOP_K_RESULTS, HF_MAX_NEW_TOKENS


@dataclass
class RAGResult:
    """Result from a RAG query."""
    answer: str
    source_chunks: list  # list of TextChunk
    context_used: str
    error: Optional[str] = None


def retrieve_context(
    query: str,
    vector_store: VectorStore,
    top_k: int = TOP_K_RESULTS,
    embedding_model=None,
) -> tuple[list, list[str]]:
    """
    Embed query and retrieve top-k relevant chunks.
    Returns (chunks, chunk_texts).
    """
    if not vector_store.is_ready:
        raise VectorStoreError(
            "No study material loaded. Please upload a document first."
        )

    query_embedding = embed_query(query, model=embedding_model)
    results = vector_store.search(query_embedding, top_k=top_k)

    chunks = [chunk for chunk, _score in results]
    chunk_texts = [chunk.text for chunk in chunks]
    return chunks, chunk_texts


def synthesize_direct_answer(question: str, chunk_texts: list[str]) -> str:
    """
    Extract and structure key points directly from retrieved notes
    when external LLM inference server is unreachable.
    """
    if not chunk_texts:
        return "No matching content found in your uploaded study material."

    # Extract keywords from question
    keywords = [w.lower() for w in re.findall(r"\w+", question) if len(w) > 3]
    
    # Split into clean candidate sentences
    scored_sentences: list[tuple[int, str]] = []
    seen = set()
    
    for c in chunk_texts:
        for s in re.split(r"(?<=[.!?\n])\s+", c):
            s_clean = s.strip()
            if len(s_clean) > 25 and s_clean not in seen:
                seen.add(s_clean)
                score = sum(2 for kw in keywords if kw in s_clean.lower())
                scored_sentences.append((score, s_clean))

    scored_sentences.sort(key=lambda x: x[0], reverse=True)
    top_sentences = [s for sc, s in scored_sentences[:6] if sc > 0]
    
    if not top_sentences:
        top_sentences = [s for _, s in scored_sentences[:4]]

    body = "\n\n".join(f"• **Key Concept:** {s}" for s in top_sentences)
    
    return f"""### 📖 Extracted Answer from Your Notes

{body}

---
*💡 Note: Generated using local document extractor. For AI generated prose, connect a free Gemini / Groq API key in the sidebar.*"""


def rag_answer(
    question: str,
    vector_store: VectorStore,
    embedding_model=None,
    max_new_tokens: int = HF_MAX_NEW_TOKENS,
    top_k: int = TOP_K_RESULTS,
) -> RAGResult:
    """
    Full RAG pipeline for answering a student's question from their notes.
    """
    if not question or not question.strip():
        return RAGResult(
            answer="Please enter a question.",
            source_chunks=[],
            context_used="",
            error="empty_question",
        )

    try:
        chunks, chunk_texts = retrieve_context(
            question, vector_store, top_k=top_k, embedding_model=embedding_model
        )
    except VectorStoreError as exc:
        return RAGResult(
            answer=str(exc),
            source_chunks=[],
            context_used="",
            error="vector_store_error",
        )

    context_block = build_context_block(chunk_texts)
    prompt = STUDY_ASSISTANT_PROMPT.format(
        context=context_block,
        question=question,
    )

    try:
        answer = generate(prompt, max_new_tokens=max_new_tokens)
    except LLMError as exc:
        # Graceful fallback: Synthesize answer directly from chunks
        fallback_answer = synthesize_direct_answer(question, chunk_texts)
        return RAGResult(
            answer=fallback_answer,
            source_chunks=chunks,
            context_used=context_block,
        )

    return RAGResult(
        answer=answer,
        source_chunks=chunks,
        context_used=context_block,
    )


def get_broad_context(
    vector_store: VectorStore,
    query: str = "main topics concepts definitions",
    top_k: int = 15,
    embedding_model=None,
) -> tuple[list, str]:
    """
    Retrieve a broad context for features that need a general overview
    (flashcards, quiz, quick revision, exam tomorrow).
    """
    chunks, chunk_texts = retrieve_context(
        query, vector_store, top_k=top_k, embedding_model=embedding_model
    )
    context_block = build_context_block(chunk_texts)
    return chunks, context_block
