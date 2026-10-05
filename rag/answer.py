"""Assemble the prompt (system prompt + retrieved sources + history) and call the LLM."""

from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path

from langsmith import traceable
from langsmith.wrappers import wrap_openai
from openai import OpenAI

from . import config
from .retriever import Chunk, retrieve

# --- system prompts ---------------------------------------------------------


def list_prompts() -> list[Path]:
    return sorted(config.PROMPTS_DIR.glob("*.md"))


def load_system_prompt(path: Path | str | None = None) -> str:
    """Read fresh on every call so edits to prompts/*.md apply immediately."""
    path = Path(path) if path else config.SYSTEM_PROMPT_PATH
    if not path.is_absolute():
        path = config.ROOT / path
    return path.read_text(encoding="utf-8").strip()


# --- prompt assembly --------------------------------------------------------


def format_sources(chunks: list[Chunk]) -> str:
    if not chunks:
        return "<sources>\n(no relevant sources were found in the corpus)\n</sources>"
    blocks = [f'<source id="{i}" file="{c.label}">\n{c.text}\n</source>' for i, c in enumerate(chunks, 1)]
    return "<sources>\n" + "\n\n".join(blocks) + "\n</sources>"


def build_messages(question: str, chunks: list[Chunk], history: list[dict], system_prompt: str) -> list[dict]:
    messages = [{"role": "system", "content": system_prompt}]
    # history is [{"role": "user"|"assistant", "content": str}, ...]; keep the last N turns.
    messages += history[-config.HISTORY_TURNS * 2 :]
    messages.append(
        {
            "role": "user",
            "content": f"Sources retrieved from the corpus for this question:\n\n{format_sources(chunks)}"
            f"\n\nQuestion: {question}",
        }
    )
    return messages


# --- LLM --------------------------------------------------------------------


@lru_cache(maxsize=1)
def llm_client() -> OpenAI:
    if not config.OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set. Add it to .env (see .env.example).")
    client = OpenAI(
        base_url=config.OPENROUTER_BASE_URL,
        api_key=config.OPENROUTER_API_KEY,
        default_headers={"X-Title": "RagPlayground"},
    )
    return wrap_openai(client)  # LLM calls show up in LangSmith traces when tracing is on


# Shown when the model's own safety layer blocks a reply (OpenRouter returns no text with
# finish_reason "content_filter"); without this the user would see an empty answer.
BLOCKED_TEXT = "I can't help with that request."


@traceable(name="rag_answer")
def answer(question: str, history: list[dict] | None = None, prompt_path: Path | str | None = None) -> dict:
    chunks = retrieve(question)
    messages = build_messages(question, chunks, history or [], load_system_prompt(prompt_path))
    resp = llm_client().chat.completions.create(
        model=config.LLM_MODEL, messages=messages, temperature=config.TEMPERATURE
    )
    text = resp.choices[0].message.content or ""
    return {"answer": text or BLOCKED_TEXT, "chunks": chunks}


def _reduce_stream(events: list) -> dict:
    chunks = next((e for e in events if isinstance(e, list)), [])
    text = "".join(e for e in events if isinstance(e, str))
    return {"answer": text, "sources": [c.label for c in chunks]}


@traceable(name="rag_answer", reduce_fn=_reduce_stream)
def stream_answer(
    question: str, history: list[dict] | None = None, prompt_path: Path | str | None = None
) -> Iterator[list[Chunk] | str]:
    """Yields the retrieved chunks first (a list), then the answer as text deltas."""
    chunks = retrieve(question)
    yield chunks
    messages = build_messages(question, chunks, history or [], load_system_prompt(prompt_path))
    stream = llm_client().chat.completions.create(
        model=config.LLM_MODEL, messages=messages, temperature=config.TEMPERATURE, stream=True
    )
    produced = False
    for event in stream:
        if event.choices and event.choices[0].delta.content:
            produced = True
            yield event.choices[0].delta.content
    if not produced:
        yield BLOCKED_TEXT
