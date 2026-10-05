"""
Exam Mode feature.
Generates structured exam-oriented answers for given topics and mark allocations.
"""

from __future__ import annotations

from src.rag import retrieve_context, get_broad_context
from src.vector_store import VectorStore, VectorStoreError
from src.gemma import generate, LLMError
from src.prompts import get_exam_prompt, build_context_block
from src.config import HF_MAX_NEW_TOKENS


VALID_MARKS = ["2", "5", "10"]


def generate_exam_answer(
    topic: str,
    marks: str,
    vector_store: VectorStore,
    embedding_model=None,
    top_k: int = 8,
) -> dict:
    """
    Generate a structured exam answer for a topic with specified marks.

    Returns:
        dict with keys: "answer", "sources", "error" (or None)
    """
    if not topic or not topic.strip():
        return {"answer": None, "sources": [], "error": "Please enter a topic."}

    marks = str(marks).strip()
    if marks not in VALID_MARKS:
        return {
            "answer": None,
            "sources": [],
            "error": f"Invalid marks value. Choose from: {', '.join(VALID_MARKS)}",
        }

    if not vector_store.is_ready:
        return {
            "answer": None,
            "sources": [],
            "error": "No study material loaded. Please upload a document first.",
        }

    try:
        chunks, chunk_texts = retrieve_context(
            topic, vector_store, top_k=top_k, embedding_model=embedding_model
        )
    except VectorStoreError as exc:
        return {"answer": None, "sources": [], "error": str(exc)}

    context_block = build_context_block(chunk_texts)
    prompt = get_exam_prompt(topic=topic, marks=marks, context=context_block)

    # Longer answers for 10-mark questions
    max_tokens = {
        "2": 300,
        "5": 600,
        "10": 1200,
    }.get(marks, HF_MAX_NEW_TOKENS)

    try:
        answer = generate(prompt, max_new_tokens=max_tokens)
        return {"answer": answer, "sources": chunks, "error": None}
    except LLMError as exc:
        # Graceful fallback: synthesize structured exam answer directly from notes
        fallback_answer = _synthesize_exam_fallback(topic, marks, chunk_texts)
        return {"answer": fallback_answer, "sources": chunks, "error": None}


def _synthesize_exam_fallback(topic: str, marks: str, chunk_texts: list[str]) -> str:
    """Synthesize structured exam points from retrieved chunks when LLM fails."""
    import re
    all_lines = []
    for text in chunk_texts:
        for line in text.split("\n"):
            clean = re.sub(r"\[Page \d+.*?\]", "", line).strip()
            if len(clean) > 25:
                all_lines.append(clean)
    
    # Prioritize lines mentioning topic keywords
    topic_words = set(topic.lower().split())
    topic_lines = [l for l in all_lines if any(w in l.lower() for w in topic_words)]
    selected_lines = topic_lines if topic_lines else all_lines
    
    num_pts = 2 if marks == "2" else (5 if marks == "5" else 8)
    chosen = selected_lines[:num_pts] if selected_lines else ["Refer to the attached source notes for primary definitions."]
    
    body = "\n\n".join(f"**Point {i+1}:** {p}" for i, p in enumerate(chosen))
    
    return f"""### 📝 Exam Answer ({marks} Marks)

**Topic:** {topic}

{body}

---
*💡 Note: Synthesized directly from your uploaded study notes.*"""

