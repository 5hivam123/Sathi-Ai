"""
SaathiAI Configuration
Centralizes all configurable settings and loads from environment variables.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Project Root ─────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# ── Document Processing ───────────────────────────────────────────────────────
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "50"))
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

# ── Embeddings ────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

# ── Vector Store ──────────────────────────────────────────────────────────────
FAISS_INDEX_PATH = DATA_DIR / "faiss_index"
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "5"))

# ── LLM Configuration ─────────────────────────────────────────────────────────
# Inference backend: "huggingface" | "ollama" | "openai_compatible"
LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama")

# HuggingFace local inference
HF_MODEL_ID = os.getenv("HF_MODEL_ID", "google/gemma-2b-it")
HF_MAX_NEW_TOKENS = int(os.getenv("HF_MAX_NEW_TOKENS", "1024"))
HF_TEMPERATURE = float(os.getenv("HF_TEMPERATURE", "0.3"))
HF_DEVICE = os.getenv("HF_DEVICE", "cpu")  # "cuda" or "cpu"

# Ollama backend (for local self-hosted inference)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma2:2b")

# OpenAI-compatible backend (e.g., LM Studio, llama.cpp server, Together AI)
OPENAI_COMPAT_BASE_URL = os.getenv("OPENAI_COMPAT_BASE_URL", "")
OPENAI_COMPAT_API_KEY = os.getenv("OPENAI_COMPAT_API_KEY", "")
OPENAI_COMPAT_MODEL = os.getenv("OPENAI_COMPAT_MODEL", "gemma-2b-it")

# Gemini API (Google AI Studio)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

# Groq Cloud API (Ultra-fast open-weight models)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# Jev AI API (TypeSafe AI / Custom Gateway)
JEV_API_KEY = os.getenv("JEV_API_KEY", "sk_jev-ai_a_orq1I0PYvgBe0TrYKNCOjxSEhByk2A")
JEV_BASE_URL = os.getenv("JEV_BASE_URL", "https://api.typesafe.ai/v1")
JEV_MODEL = os.getenv("JEV_MODEL", "jev-latest")

# HuggingFace Inference API (fallback / cloud)
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")
HF_API_MODEL = os.getenv("HF_API_MODEL", "google/gemma-2-2b-it")

# ── Admin Security ────────────────────────────────────────────────────────────
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")

# ── SQLite ────────────────────────────────────────────────────────────────────
DB_PATH = DATA_DIR / "saathiai.db"

# ── UI ────────────────────────────────────────────────────────────────────────
APP_NAME = "SaathiAI"
APP_TAGLINE = "Turn your notes into your personal AI tutor."
APP_VERSION = "1.0.0"

ALLOWED_FILE_TYPES = ["pdf", "txt"]

