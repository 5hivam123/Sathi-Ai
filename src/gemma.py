"""
LLM integration module.
Supports multiple inference backends:
  - ollama       : local Ollama server (e.g. gemma2:2b, llama3)
  - gemini       : Google Gemini API (gemini-1.5-flash, gemini-2.0-flash)
  - groq         : Groq Cloud API (gemma2-9b-it, llama-3.3-70b)
  - openai_compat: any OpenAI-compatible API (LM Studio, Together AI, OpenAI)
  - huggingface  : local HuggingFace transformers pipeline
  - hf_api       : HuggingFace Inference API (cloud fallback)

Configure via environment variables (see .env.example) or directly in the UI.
"""

from __future__ import annotations

import os
import requests
from typing import Optional

from src.config import (
    LLM_BACKEND,
    HF_MODEL_ID,
    HF_MAX_NEW_TOKENS,
    HF_TEMPERATURE,
    HF_DEVICE,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GROQ_API_KEY,
    GROQ_MODEL,
    JEV_API_KEY,
    JEV_BASE_URL,
    JEV_MODEL,
    OPENAI_COMPAT_BASE_URL,
    OPENAI_COMPAT_API_KEY,
    OPENAI_COMPAT_MODEL,
    HF_API_TOKEN,
    HF_API_MODEL,
)


class LLMError(Exception):
    """Raised when LLM inference fails."""


# ── Backend implementations ───────────────────────────────────────────────────

def _generate_gemini(prompt: str, max_new_tokens: int) -> str:
    """Generate using Google Gemini API."""
    api_key = os.getenv("GEMINI_API_KEY", GEMINI_API_KEY)
    if not api_key:
        raise LLMError(
            "GEMINI_API_KEY is not set. Get a free API key at https://aistudio.google.com/ "
            "and add it to your .env file as GEMINI_API_KEY=your_key"
        )

    model = os.getenv("GEMINI_MODEL", GEMINI_MODEL) or "gemini-1.5-flash"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "maxOutputTokens": max_new_tokens,
            "temperature": HF_TEMPERATURE,
        },
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        candidates = data.get("candidates", [])
        if not candidates:
            return ""
        parts = candidates[0].get("content", {}).get("parts", [])
        if parts:
            return parts[0].get("text", "").strip()
        return ""
    except requests.exceptions.HTTPError as exc:
        raise LLMError(f"Gemini API Error: {resp.text}") from exc
    except Exception as exc:
        raise LLMError(f"Gemini request failed: {exc}") from exc


def _generate_groq(prompt: str, max_new_tokens: int) -> str:
    """Generate using Groq Cloud API (super fast open-weight models)."""
    api_key = os.getenv("GROQ_API_KEY", GROQ_API_KEY)
    if not api_key:
        raise LLMError(
            "GROQ_API_KEY is not set. Get a free API key at https://console.groq.com/ "
            "and add it to your .env file as GROQ_API_KEY=your_key"
        )

    model = os.getenv("GROQ_MODEL", GROQ_MODEL) or "gemma2-9b-it"
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_new_tokens,
        "temperature": HF_TEMPERATURE,
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except requests.exceptions.HTTPError as exc:
        raise LLMError(f"Groq API Error: {resp.text}") from exc
    except Exception as exc:
        raise LLMError(f"Groq request failed: {exc}") from exc


def _generate_ollama(prompt: str, max_new_tokens: int) -> str:
    """Generate using a local Ollama server."""
    try:
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_new_tokens,
                "temperature": HF_TEMPERATURE,
            },
        }
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()

    except Exception as exc:
        raise LLMError(
            f"Ollama inference failed. Make sure Ollama is running at {OLLAMA_BASE_URL} "
            f"and model '{OLLAMA_MODEL}' is pulled (`ollama run {OLLAMA_MODEL}`). Error: {exc}"
        ) from exc


def _generate_openai_compat(prompt: str, max_new_tokens: int) -> str:
    """Generate using any OpenAI-compatible API endpoint."""
    base_url = os.getenv("OPENAI_COMPAT_BASE_URL", OPENAI_COMPAT_BASE_URL) or "https://api.openai.com/v1"
    api_key = os.getenv("OPENAI_COMPAT_API_KEY", OPENAI_COMPAT_API_KEY) or "not-needed"
    model = os.getenv("OPENAI_COMPAT_MODEL", OPENAI_COMPAT_MODEL) or "gpt-4o-mini"

    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_new_tokens,
        "temperature": HF_TEMPERATURE,
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=90)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        raise LLMError(f"OpenAI-compatible API call failed: {exc}") from exc


def _generate_huggingface(prompt: str, max_new_tokens: int) -> str:
    """Generate using a local HuggingFace transformers pipeline."""
    try:
        import torch
        from transformers import pipeline

        if not hasattr(_generate_huggingface, "_pipe"):
            device = 0 if (HF_DEVICE == "cuda" and torch.cuda.is_available()) else -1
            _generate_huggingface._pipe = pipeline(
                "text-generation",
                model=HF_MODEL_ID,
                device=device,
                torch_dtype=torch.float16 if device == 0 else torch.float32,
                trust_remote_code=True,
            )

        pipe = _generate_huggingface._pipe
        result = pipe(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=HF_TEMPERATURE,
            do_sample=HF_TEMPERATURE > 0,
            pad_token_id=pipe.tokenizer.eos_token_id,
            return_full_text=False,
        )
        return result[0]["generated_text"].strip()
    except Exception as exc:
        raise LLMError(f"HuggingFace inference failed: {exc}") from exc


def _generate_hf_api(prompt: str, max_new_tokens: int) -> str:
    """Generate using the HuggingFace Inference API (cloud)."""
    if not HF_API_TOKEN:
        raise LLMError(
            "HF_API_TOKEN is not set. Get a token from huggingface.co/settings/tokens."
        )
    try:
        headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_new_tokens,
                "temperature": HF_TEMPERATURE,
                "return_full_text": False,
            },
        }
        url = f"https://api-inference.huggingface.co/models/{HF_API_MODEL}"
        resp = requests.post(url, headers=headers, json=payload, timeout=120)

        if resp.status_code == 503:
            raise LLMError(f"The model '{HF_API_MODEL}' is currently loading on HuggingFace. Retry shortly.")
        resp.raise_for_status()
        data = resp.json()

        if isinstance(data, list) and data:
            return data[0].get("generated_text", "").strip()
        elif isinstance(data, dict) and "error" in data:
            raise LLMError(f"HF API error: {data['error']}")
        return str(data).strip()
    except Exception as exc:
        raise LLMError(f"HuggingFace API call failed: {exc}") from exc


def _generate_jev(prompt: str, max_new_tokens: int) -> str:
    """Generate using Jev AI (TypeSafe AI / Custom Gateway)."""
    api_key = os.getenv("JEV_API_KEY", JEV_API_KEY)
    if not api_key:
        raise LLMError("JEV_API_KEY is not set.")

    base_url = os.getenv("JEV_BASE_URL", JEV_BASE_URL) or "https://api.typesafe.ai/v1"
    model = os.getenv("JEV_MODEL", JEV_MODEL) or "jev-latest"

    # Try standard chat completions endpoint
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_new_tokens,
        "temperature": HF_TEMPERATURE,
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        if resp.status_code == 200:
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        elif resp.status_code == 401:
            raise LLMError(f"Jev AI Authentication failed (401): {resp.text}")
        resp.raise_for_status()
    except requests.exceptions.RequestException as exc:
        raise LLMError(f"Jev AI API request failed: {exc}") from exc


# ── Public Interface ──────────────────────────────────────────────────────────

def generate(
    prompt: str,
    max_new_tokens: int = HF_MAX_NEW_TOKENS,
    backend: Optional[str] = None,
) -> str:
    """
    Generate text using the configured LLM backend.
    """
    chosen_backend = (backend or os.getenv("LLM_BACKEND", LLM_BACKEND)).lower()

    if chosen_backend in ("jev", "jev_ai", "jev-ai"):
        return _generate_jev(prompt, max_new_tokens)
    elif chosen_backend == "gemini":
        return _generate_gemini(prompt, max_new_tokens)
    elif chosen_backend == "groq":
        return _generate_groq(prompt, max_new_tokens)
    elif chosen_backend == "ollama":
        return _generate_ollama(prompt, max_new_tokens)
    elif chosen_backend in ("openai_compat", "openai-compat", "openai"):
        return _generate_openai_compat(prompt, max_new_tokens)
    elif chosen_backend == "huggingface":
        return _generate_huggingface(prompt, max_new_tokens)
    elif chosen_backend in ("hf_api", "hf-api"):
        return _generate_hf_api(prompt, max_new_tokens)
    else:
        if os.getenv("JEV_API_KEY"):
            return _generate_jev(prompt, max_new_tokens)
        elif os.getenv("GEMINI_API_KEY"):
            return _generate_gemini(prompt, max_new_tokens)
        return _generate_ollama(prompt, max_new_tokens)


def get_backend_display_name() -> str:
    """Return a human-readable description of the current inference backend."""
    backend = os.getenv("LLM_BACKEND", LLM_BACKEND).lower()
    if backend in ("jev", "jev_ai", "jev-ai"):
        return f"Jev AI ({os.getenv('JEV_MODEL', JEV_MODEL)})"
    elif backend == "gemini":
        return f"Google Gemini API ({os.getenv('GEMINI_MODEL', GEMINI_MODEL)})"
    elif backend == "groq":
        return f"Groq Cloud ({os.getenv('GROQ_MODEL', GROQ_MODEL)})"
    elif backend == "ollama":
        return f"Ollama Local ({OLLAMA_MODEL})"
    elif backend in ("openai_compat", "openai-compat", "openai"):
        return f"OpenAI / Custom API ({OPENAI_COMPAT_MODEL})"
    elif backend == "huggingface":
        return f"Local HuggingFace ({HF_MODEL_ID})"
    elif backend in ("hf_api", "hf-api"):
        return f"HuggingFace API ({HF_API_MODEL})"
    return backend

