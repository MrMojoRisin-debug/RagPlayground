"""Central settings. Every value can be overridden from .env (see .env.example)."""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _env(name: str, default: str) -> str:
    return os.getenv(name) or default


# Paths
CORPUS_DIR = ROOT / "corpus"
DATA_DIR = ROOT / "data"
CHROMA_DIR = DATA_DIR / "chroma"
MANIFEST_PATH = DATA_DIR / "manifest.json"
PROMPTS_DIR = ROOT / "prompts"
EVAL_DATASET_PATH = ROOT / "evals" / "dataset.jsonl"
COLLECTION_NAME = "corpus"

# LLM via OpenRouter (OpenAI-compatible API)
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
LLM_MODEL = _env("RAG_LLM_MODEL", "anthropic/claude-sonnet-5.5")
JUDGE_MODEL = _env("RAG_JUDGE_MODEL", "anthropic/claude-sonnet-5.5")
TEMPERATURE = float(_env("RAG_TEMPERATURE", "0.1"))

# System prompt
SYSTEM_PROMPT_PATH = ROOT / _env("RAG_SYSTEM_PROMPT", "prompts/system.md")
REFUSAL_TEXT = _env("RAG_REFUSAL_TEXT", "I couldn't find that in the corpus.")

# Embeddings + chunking (sizes are in characters; ~4 chars per token)
EMBED_MODEL = _env("RAG_EMBED_MODEL", "Qwen/Qwen3-Embedding-0.6B")
CHUNK_SIZE = int(_env("RAG_CHUNK_SIZE", "3200"))
CHUNK_OVERLAP = int(_env("RAG_CHUNK_OVERLAP", "400"))

# Retrieval / chat
TOP_K = int(_env("RAG_TOP_K", "6"))
HISTORY_TURNS = int(_env("RAG_HISTORY_TURNS", "3"))

# LangSmith
LANGSMITH_DATASET = _env("LANGSMITH_DATASET", "rag-playground-eval")
