"""
Flashcard generator module.
Creates front/back flashcards from study material using LLM with JSON output.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from src.rag import get_broad_context, retrieve_context
from src.vector_store import VectorStore
from src.gemma import generate, LLMError
from src.prompts import get_flashcard_prompt, build_context_block


@dataclass
class Flashcard:
    front: str  # Question or term
    back: str   # Answer or definition
    index: int = 0


@dataclass
class FlashcardDeck:
    cards: list[Flashcard]
    error: str = ""

    @property
    def is_valid(self) -> bool:
        return len(self.cards) > 0 and not self.error

    @property
    def count(self) -> int:
        return len(self.cards)


def generate_flashcards(
    vector_store: VectorStore,
    num_cards: int = 10,
    topic: str = "",
    embedding_model=None,
) -> FlashcardDeck:
    """
    Generate flashcards from uploaded study material.

    Args:
        vector_store: The loaded FAISS vector store.
        num_cards: Number of flashcards to generate.
        topic: Optional topic to focus on.
        embedding_model: Sentence transformer model.

    Returns:
        FlashcardDeck with cards or error message.
    """
    if not vector_store.is_ready:
        return FlashcardDeck(
            cards=[], error="No study material loaded. Please upload a document first."
        )

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
        return FlashcardDeck(cards=[], error=f"Failed to retrieve context: {exc}")

    prompt = get_flashcard_prompt(context=context_block, num_cards=num_cards)

    try:
        raw_output = generate(prompt, max_new_tokens=1500)
        deck = _parse_flashcard_output(raw_output)
        if deck.is_valid:
            return deck
    except LLMError:
        pass

    # Graceful fallback: generate flashcards directly from document context
    fallback_cards = _generate_fallback_flashcards(context_block, num_cards)
    if fallback_cards:
        return FlashcardDeck(cards=fallback_cards)

    return FlashcardDeck(
        cards=[],
        error="Unable to generate flashcards. Please connect a free Gemini or Groq API key in the sidebar.",
    )


def _generate_fallback_flashcards(context_block: str, num_cards: int) -> list[Flashcard]:
    """Generate structured front/back flashcards directly from document sentences."""
    clean_text = re.sub(r"\[Page \d+.*?\]", "", context_block)
    raw_sentences = [s.strip() for s in re.split(r"[.\n]+", clean_text) if len(s.strip()) > 30]
    cards = []
    for i, sent in enumerate(raw_sentences[:num_cards]):
        words = sent.split()
        if len(words) < 4:
            continue
        term = " ".join(words[:3])
        definition = " ".join(words[3:])
        cards.append(Flashcard(
            front=f"What is {term}?",
            back=definition[:140] + ("…" if len(definition) > 140 else ""),
            index=i,
        ))
    return cards



# ── Parsing ───────────────────────────────────────────────────────────────────

def _parse_flashcard_output(raw_output: str) -> FlashcardDeck:
    """Parse flashcard JSON from LLM output with multiple strategies."""
    # Strategy 1: Direct JSON parse
    cards = _try_parse_json(raw_output)
    if cards:
        return FlashcardDeck(cards=cards)

    # Strategy 2: Find JSON array
    match = re.search(r"\[.*\]", raw_output, re.DOTALL)
    if match:
        cards = _try_parse_json(match.group())
        if cards:
            return FlashcardDeck(cards=cards)

    # Strategy 3: Manual extraction of front/back pairs
    fronts = re.findall(r'"front"\s*:\s*"([^"]+)"', raw_output)
    backs = re.findall(r'"back"\s*:\s*"([^"]+)"', raw_output)
    if fronts and backs and len(fronts) == len(backs):
        cards = [
            Flashcard(front=f, back=b, index=i)
            for i, (f, b) in enumerate(zip(fronts, backs))
        ]
        return FlashcardDeck(cards=cards)

    return FlashcardDeck(
        cards=[],
        error=(
            "Could not parse flashcards from model output. "
            "Please try again — smaller models occasionally produce malformed JSON."
        ),
    )


def _try_parse_json(text: str) -> list[Flashcard]:
    text = text.strip()
    text = re.sub(r"```(?:json)?", "", text).strip()

    try:
        data = json.loads(text)
        if isinstance(data, list):
            cards = []
            for i, item in enumerate(data):
                if isinstance(item, dict) and "front" in item and "back" in item:
                    cards.append(
                        Flashcard(
                            front=str(item["front"]).strip(),
                            back=str(item["back"]).strip(),
                            index=i,
                        )
                    )
            return cards
    except (json.JSONDecodeError, TypeError):
        pass
    return []
