"""
Quick Revision and Exam Tomorrow Mode modules.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from src.rag import retrieve_context, get_broad_context
from src.vector_store import VectorStore
from src.gemma import generate, LLMError
from src.prompts import (
    get_revision_prompt,
    get_exam_tomorrow_topics_prompt,
    get_exam_tomorrow_content_prompt,
    build_context_block,
)


# ── Quick Revision ────────────────────────────────────────────────────────────

@dataclass
class RevisionResult:
    content: str  # Full markdown content
    sources: list = field(default_factory=list)
    error: str = ""

    @property
    def is_valid(self) -> bool:
        return bool(self.content) and not self.error


def generate_quick_revision(
    topic: str,
    vector_store: VectorStore,
    embedding_model=None,
    top_k: int = 10,
) -> RevisionResult:
    """Generate a quick revision guide for a given topic."""
    if not topic or not topic.strip():
        return RevisionResult(content="", error="Please enter a topic to revise.")

    if not vector_store.is_ready:
        return RevisionResult(
            content="", error="No study material loaded. Please upload a document first."
        )

    try:
        chunks, chunk_texts = retrieve_context(
            topic, vector_store, top_k=top_k, embedding_model=embedding_model
        )
    except Exception as exc:
        return RevisionResult(content="", error=f"Context retrieval failed: {exc}")

    if not chunk_texts:
        return RevisionResult(
            content="",
            error=f"No relevant content found for topic '{topic}' in your notes.",
        )

    context_block = build_context_block(chunk_texts)
    prompt = get_revision_prompt(topic=topic, context=context_block)
    try:
        content = generate(prompt, max_new_tokens=1500)
        return RevisionResult(content=content, sources=chunks)
    except LLMError as exc:
        # Graceful fallback: synthesize quick revision directly from notes
        fallback_content = _synthesize_quick_revision_fallback(topic, chunk_texts)
        return RevisionResult(content=fallback_content, sources=chunks)


def _synthesize_quick_revision_fallback(topic: str, chunk_texts: list[str]) -> str:
    """Synthesize structured quick revision notes directly from retrieved chunks."""
    clean_lines = []
    for text in chunk_texts:
        for line in text.split("\n"):
            clean = re.sub(r"\[Page \d+.*?\]", "", line).strip()
            if len(clean) > 20:
                clean_lines.append(clean)
    
    # Select distinct concepts
    concepts = clean_lines[:6] if clean_lines else ["Review primary sections in your notes."]
    points = "\n".join(f"- **Key Fact:** {c}" for c in concepts)
    
    return f"""## ⚡ Quick Revision: {topic}

### 📌 Core Concepts & High-Yield Definitions
{points}

### 🎯 Quick Summary
- Focus on the main terminology and definitions highlighted above.
- Review related numerical examples and theoretical proofs from your syllabus.

---
*💡 Note: Synthesized directly from your uploaded study notes.*"""



# ── Exam Tomorrow Mode ────────────────────────────────────────────────────────

@dataclass
class ImportantTopic:
    rank: int
    topic: str
    reason: str


@dataclass
class ExamTomorrowResult:
    important_topics: list[ImportantTopic] = field(default_factory=list)
    revision_content: str = ""
    error: str = ""

    @property
    def is_valid(self) -> bool:
        return bool(self.revision_content) and not self.error


def generate_exam_tomorrow_plan(
    vector_store: VectorStore,
    embedding_model=None,
    num_topics: int = 8,
) -> ExamTomorrowResult:
    """
    Generate the complete Exam Tomorrow emergency revision package.

    Phase 1: Identify important topics (JSON)
    Phase 2: Generate full revision content using those topics

    Returns ExamTomorrowResult with topics and formatted content.
    """
    if not vector_store.is_ready:
        return ExamTomorrowResult(
            error="No study material loaded. Please upload a document first."
        )

    # ── Phase 1: Identify important topics ───────────────────────────────────
    try:
        _, context_block = get_broad_context(
            vector_store, top_k=20, embedding_model=embedding_model
        )
    except Exception as exc:
        return ExamTomorrowResult(error=f"Failed to retrieve context: {exc}")

    topics_prompt = get_exam_tomorrow_topics_prompt(
        context=context_block, num_topics=num_topics
    )
    important_topics = []
    try:
        topics_raw = generate(topics_prompt, max_new_tokens=800)
        important_topics = _parse_topics(topics_raw, num_topics)
    except LLMError:
        # Fallback topics from context headings/key phrases
        important_topics = _fallback_important_topics(context_block, num_topics)

    topics_list = "\n".join(
        f"- {t.topic}" for t in important_topics
    ) if important_topics else "Core study concepts"

    # ── Phase 2: Generate full revision content ───────────────────────────────
    content_prompt = get_exam_tomorrow_content_prompt(
        context=context_block,
        topics_list=topics_list,
    )

    try:
        revision_content = generate(content_prompt, max_new_tokens=3000)
        return ExamTomorrowResult(
            important_topics=important_topics,
            revision_content=revision_content,
        )
    except LLMError:
        fallback_content = _fallback_exam_tomorrow_content(context_block, important_topics)
        return ExamTomorrowResult(
            important_topics=important_topics,
            revision_content=fallback_content,
        )


def _fallback_important_topics(context_block: str, num_topics: int) -> list[ImportantTopic]:
    clean_text = re.sub(r"\[Page \d+.*?\]", "", context_block)
    lines = [s.strip() for s in re.split(r"[.\n]+", clean_text) if len(s.strip()) > 30]
    topics = []
    for i, line in enumerate(lines[:num_topics]):
        words = line.split()
        if len(words) < 4:
            continue
        topic_name = " ".join(words[:4])
        topics.append(ImportantTopic(
            rank=i + 1,
            topic=topic_name,
            reason=f"Foundational topic from study notes: {line[:80]}…",
        ))
    return topics


def _fallback_exam_tomorrow_content(context_block: str, topics: list[ImportantTopic]) -> str:
    topic_summary = "\n".join(f"{t.rank}. **{t.topic}**: {t.reason}" for t in topics)
    return f"""# 🚨 Emergency Exam Revision Package

## 🎯 Top High-Yield Topics to Focus On
{topic_summary}

---

## 📌 Key Definitions & Concepts
- Review the core definitions and properties of the topics listed above.
- Ensure clear understanding of main theoretical concepts directly from the uploaded material.

---
*💡 Note: Generated using direct document extraction fallback.*"""



def _parse_topics(raw: str, num_topics: int) -> list[ImportantTopic]:
    """Parse the topics JSON output from the LLM."""
    raw = raw.strip()
    raw = re.sub(r"```(?:json)?", "", raw).strip()

    # Try direct JSON parse
    for pattern in [raw, _extract_json_array(raw)]:
        if not pattern:
            continue
        try:
            data = json.loads(pattern)
            if isinstance(data, list):
                topics = []
                for item in data:
                    if isinstance(item, dict) and "topic" in item:
                        topics.append(
                            ImportantTopic(
                                rank=int(item.get("rank", len(topics) + 1)),
                                topic=str(item["topic"]).strip(),
                                reason=str(item.get("reason", "")).strip(),
                            )
                        )
                if topics:
                    return topics[:num_topics]
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    # Fallback: extract topic names from text
    lines = [line.strip() for line in raw.split("\n") if line.strip()]
    topics = []
    for i, line in enumerate(lines[:num_topics], start=1):
        line = re.sub(r'^[\d\.\-\*]+\s*', '', line)
        if line:
            topics.append(ImportantTopic(rank=i, topic=line, reason=""))
    return topics


def _extract_json_array(text: str) -> str:
    match = re.search(r"\[.*\]", text, re.DOTALL)
    return match.group() if match else ""
