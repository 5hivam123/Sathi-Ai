"""
Quiz Generator module.
Generates MCQ quizzes from study material using LLM with structured JSON output.
Includes robust parsing with fallback strategies.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from src.rag import get_broad_context, retrieve_context
from src.vector_store import VectorStore
from src.gemma import generate, LLMError
from src.prompts import get_quiz_prompt, build_context_block


@dataclass
class QuizQuestion:
    question: str
    options: dict  # {"A": "...", "B": "...", "C": "...", "D": "..."}
    correct_answer: str  # "A", "B", "C", or "D"
    explanation: str
    user_answer: str = ""  # filled when student submits

    @property
    def is_correct(self) -> bool:
        return self.user_answer.upper() == self.correct_answer.upper()


@dataclass
class QuizResult:
    questions: list[QuizQuestion] = field(default_factory=list)
    error: str = ""

    @property
    def is_valid(self) -> bool:
        return len(self.questions) > 0 and not self.error

    @property
    def total(self) -> int:
        return len(self.questions)

    def score(self) -> tuple[int, int]:
        """Returns (correct_count, total_count)."""
        answered = [q for q in self.questions if q.user_answer]
        correct = sum(1 for q in answered if q.is_correct)
        return correct, len(answered)


def generate_quiz(
    vector_store: VectorStore,
    num_questions: int = 10,
    difficulty: str = "Medium",
    topic: str = "",
    embedding_model=None,
) -> QuizResult:
    """
    Generate a multiple-choice quiz from the uploaded study material.

    Args:
        vector_store: The loaded FAISS vector store.
        num_questions: Number of MCQ questions to generate (5, 10, or 15).
        difficulty: "Easy", "Medium", or "Hard".
        topic: Optional topic to focus the quiz on.
        embedding_model: The sentence transformer model.

    Returns:
        QuizResult with parsed questions or an error message.
    """
    if not vector_store.is_ready:
        return QuizResult(error="No study material loaded. Please upload a document first.")

    try:
        if topic.strip():
            chunks, chunk_texts = retrieve_context(
                topic, vector_store, top_k=12, embedding_model=embedding_model
            )
            context_block = build_context_block(chunk_texts)
        else:
            _, context_block = get_broad_context(
                vector_store, top_k=12, embedding_model=embedding_model
            )
    except Exception as exc:
        return QuizResult(error=f"Failed to retrieve context: {exc}")

    prompt = get_quiz_prompt(
        context=context_block,
        num_questions=num_questions,
        difficulty=difficulty,
    )

    try:
        raw_output = generate(prompt, max_new_tokens=2000)
        parsed = _parse_quiz_output(raw_output, num_questions)
        if parsed.is_valid:
            return parsed
    except LLMError:
        pass

    # Graceful fallback: build quiz questions directly from document concepts
    fallback_qs = _generate_fallback_quiz_from_context(context_block, num_questions)
    if fallback_qs:
        return QuizResult(questions=fallback_qs)

    return QuizResult(error="Unable to generate quiz. Please connect a free Gemini or Groq API key in the sidebar.")


def _generate_fallback_quiz_from_context(context_block: str, num_questions: int) -> list[QuizQuestion]:
    """Extract key sentences from document context to build a fallback quiz."""
    clean_text = re.sub(r"\[Page \d+.*?\]", "", context_block)
    raw_sentences = [s.strip() for s in re.split(r"[.\n]+", clean_text) if len(s.strip()) > 35]
    
    questions = []
    for i, sent in enumerate(raw_sentences[:num_questions]):
        words = sent.split()
        if len(words) < 5:
            continue
        term = " ".join(words[:3])
        definition = " ".join(words[3:])
        
        q_text = f"According to your uploaded notes, what is described regarding '{term}'?"
        opts = {
            "A": definition[:100] + ("…" if len(definition) > 100 else ""),
            "B": "It operates in reverse order and is omitted from standard evaluation.",
            "C": "It is an unverified hypothesis not covered in this curriculum.",
            "D": "None of the above statements apply accurately.",
        }
        questions.append(QuizQuestion(
            question=q_text,
            options=opts,
            correct_answer="A",
            explanation=f"Referenced directly from your study material: '{sent[:120]}…'",
        ))
    return questions


# ── JSON Parsing with Fallbacks ───────────────────────────────────────────────

def _parse_quiz_output(raw_output: str, expected_count: int = 10) -> QuizResult:
    """
    Attempt to parse quiz JSON from LLM output.
    Uses multiple strategies to handle imperfect model outputs.
    """
    # Strategy 1: Find JSON array directly
    questions = _try_parse_json(raw_output)
    if questions:
        return QuizResult(questions=questions)

    # Strategy 2: Extract content between first [ and last ]
    match = re.search(r"\[.*\]", raw_output, re.DOTALL)
    if match:
        questions = _try_parse_json(match.group())
        if questions:
            return QuizResult(questions=questions)

    # Strategy 3: Extract individual JSON objects
    question_blocks = re.findall(
        r'\{[^{}]*"question"[^{}]*"options"[^{}]*"correct_answer"[^{}]*\}',
        raw_output,
        re.DOTALL,
    )
    if question_blocks:
        questions = []
        for block in question_blocks:
            q = _try_parse_single(block)
            if q:
                questions.append(q)
        if questions:
            return QuizResult(questions=questions)

    return QuizResult(
        error=(
            "Could not parse quiz questions from the model output. "
            "This sometimes happens with smaller models. Please try again."
        )
    )


def _try_parse_json(text: str) -> list[QuizQuestion]:
    """Try to parse a JSON array of quiz questions."""
    text = text.strip()
    # Remove markdown code fences
    text = re.sub(r"```(?:json)?", "", text).strip()

    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [_dict_to_question(item) for item in data if _is_valid_q(item)]
    except (json.JSONDecodeError, KeyError, TypeError):
        pass
    return []


def _try_parse_single(text: str) -> QuizQuestion | None:
    """Try to parse a single question JSON object."""
    try:
        data = json.loads(text)
        if _is_valid_q(data):
            return _dict_to_question(data)
    except (json.JSONDecodeError, KeyError, TypeError):
        pass
    return None


def _is_valid_q(item: dict) -> bool:
    return (
        isinstance(item, dict)
        and "question" in item
        and "options" in item
        and "correct_answer" in item
        and isinstance(item.get("options"), dict)
        and len(item.get("options", {})) >= 4
    )


def _dict_to_question(data: dict) -> QuizQuestion:
    return QuizQuestion(
        question=str(data["question"]),
        options={k: str(v) for k, v in data["options"].items()},
        correct_answer=str(data["correct_answer"]).upper().strip(),
        explanation=str(data.get("explanation", "")),
    )
