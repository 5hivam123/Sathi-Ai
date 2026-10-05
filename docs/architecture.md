# SaathiAI Architecture

## System Overview

SaathiAI implements a Retrieval-Augmented Generation (RAG) pipeline to ground LLM responses in uploaded study material.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         DOCUMENT INGESTION                       │
│                                                                   │
│  PDF / TXT Upload                                                 │
│       ↓                                                           │
│  File Validation (type, size, content check)                      │
│       ↓                                                           │
│  Text Extraction (PyMuPDF for PDF, unicode decode for TXT)        │
│       ↓                                                           │
│  Text Cleaning (unicode normalize, control char removal)          │
│       ↓                                                           │
│  Paragraph-aware Chunking with Overlap                            │
│  (chunk_size=800 chars, overlap=150 chars)                        │
│       ↓                                                           │
│  Sentence Transformer Embedding                                   │
│  (all-MiniLM-L6-v2, 384-dim vectors)                             │
│       ↓                                                           │
│  FAISS IndexFlatIP (cosine similarity via normalized vectors)     │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                         QUERY / GENERATION                       │
│                                                                   │
│  Student Query / Feature Request                                  │
│       ↓                                                           │
│  Query Embedding (same Sentence Transformer)                      │
│       ↓                                                           │
│  FAISS Similarity Search (top-k chunks)                           │
│       ↓                                                           │
│  Context Construction (chunks → formatted context block)          │
│       ↓                                                           │
│  Feature-specific Prompt Template (src/prompts.py)               │
│       ↓                                                           │
│  Gemma (or other open-weight LLM via chosen backend)             │
│       ↓                                                           │
│  Grounded Response                                                │
│       ↓                                                           │
│  Streamlit UI Rendering                                           │
└─────────────────────────────────────────────────────────────────┘
```

## Module Descriptions

### `src/config.py`
Central configuration. All parameters loaded from environment variables with sensible defaults.

### `src/pdf_processor.py`
- File validation (type, size, PDF header check)
- PDF text extraction using PyMuPDF with per-page metadata
- TXT extraction with encoding fallback
- Text cleaning: unicode normalization, null byte removal, blank line collapsing

### `src/text_splitter.py`
- Paragraph-aware chunking to preserve semantic units
- Sentence-boundary splitting for oversized paragraphs
- Overlap between chunks to avoid context loss at boundaries
- Per-page chunk metadata for source citations

### `src/embeddings.py`
- Loads Sentence Transformer model (cached at module level)
- Generates normalized embeddings for batch indexing and query

### `src/vector_store.py`
- FAISS IndexFlatIP for cosine similarity (via L2-normalized embeddings)
- Stores TextChunk metadata alongside index
- Persist/load to disk for resumable sessions

### `src/gemma.py`
- Pluggable inference backends:
  - `huggingface`: local transformers pipeline
  - `ollama`: local Ollama server REST API
  - `openai_compat`: any OpenAI-compatible API
  - `hf_api`: HuggingFace Inference API
- Consistent `generate(prompt)` interface regardless of backend

### `src/prompts.py`
- All prompt templates centralized
- Each prompt instructs the model to stay grounded in context
- Structured output prompts for quiz and flashcard JSON

### `src/rag.py`
- `rag_answer()`: Full RAG pipeline for Q&A
- `retrieve_context()`: Embedding + FAISS search
- `get_broad_context()`: Wide retrieval for feature generation

### `src/exam_mode.py`
- Structured answer generation for 2/5/10 marks
- Context retrieval focused on the specified topic
- Token budget adjusted by marks

### `src/quiz_generator.py`
- MCQ generation with JSON output format
- Multi-strategy JSON parsing with regex fallbacks
- QuizResult and QuizQuestion dataclasses

### `src/flashcards.py`
- Front/back flashcard generation with JSON output
- FlashcardDeck and Flashcard dataclasses

### `src/revision.py`
- Quick revision guide generation
- Exam Tomorrow Mode: two-phase generation (topic identification → full package)
- ExamTomorrowResult with important topics and revision content

## LLM Backend Decision

SaathiAI defaults to **Ollama** as the recommended backend because:

1. **Free and local**: No API costs, no data leaving your machine.
2. **Simple setup**: `ollama pull gemma2:2b` gets you started.
3. **Reasonable hardware**: gemma2:2b runs on most modern laptops with 8GB RAM.
4. **Swappable**: The `LLM_BACKEND` env var lets you switch without code changes.

For cloud deployment (Render free tier), the **HuggingFace Inference API** backend is recommended since Render free instances don't have enough RAM for local model loading.

## Deployment Considerations

| Deployment | Recommended Backend | RAM Requirement |
|---|---|---|
| Local development | `ollama` | 8 GB RAM |
| Local with GPU | `huggingface` | 8 GB VRAM |
| Render free tier | `hf_api` | < 512 MB |
| LM Studio | `openai_compat` | 8 GB RAM |
| VPS with GPU | `huggingface` | 8 GB VRAM |
