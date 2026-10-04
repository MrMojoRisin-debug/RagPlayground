# RAG Playground

Ask questions about your own documents. Local embeddings (Qwen3-Embedding-0.6B) + ChromaDB for
retrieval, any OpenRouter model for answers, LangSmith for evals and tracing.

## Setup

```bash
cp .env.example .env      # then fill in OPENROUTER_API_KEY (and LANGSMITH_API_KEY for evals)
uv sync                   # installs dependencies (Python 3.12 venv in .venv/)
uv run ingest             # index the sample corpus (first run downloads the ~1.2 GB embedding model)
uv run ui                 # opens the chat UI at http://localhost:8501
```

## Day-to-day

| Task | How |
|---|---|
| Add / update / remove documents | Edit files in `corpus/` (subfolders fine), then `uv run ingest` or click **Sync corpus folder** in the UI |
| Re-index everything | `uv run ingest --rebuild` (automatic if you change the embedding model or chunk settings) |
| Change guardrails or answer style | Edit `prompts/system.md` - applies on the next question, no restart |
| Try another prompt | Add `prompts/<name>.md`, pick it in the UI sidebar |
| Run an eval | `uv run eval` (or `uv run eval --prompt prompts/concise.md`) |

Supported file types: `.md .txt .rst .csv .json .pdf .docx .html`. Scanned PDFs need OCR first.

## How it works

```
corpus/ ──ingest──> chunk (~800 tokens) ──> embed (Qwen3, local) ──> ChromaDB (data/)

question ──> embed ──> top-k chunks ──> [system prompt + <sources> + history + question] ──> OpenRouter LLM
                                                                                       └─> answer with [n] citations
```

- **Incremental ingest**: `data/manifest.json` stores a content hash per file, so only new or changed
  files are re-embedded and deleted files are removed from the index.
- **System prompt** (`prompts/*.md`): your guardrails and style. Retrieved sources are appended by code in
  a fixed `<sources>` block, so the prompt file can be edited freely.
- **Refusal phrase**: the default prompt says to answer `I couldn't find that in the corpus.` when the
  sources don't cover a question. If you change that phrase, update `RAG_REFUSAL_TEXT` in `.env` too -
  the eval uses it to detect refusals.

## Evals (LangSmith)

`evals/dataset.jsonl` holds the test set, one JSON object per line:

```json
{"question": "How long is the return window?", "reference": "30 days from delivery", "expected_sources": ["policies/refunds-and-returns.md"]}
{"question": "Who won the 2022 World Cup?", "reference": "REFUSE"}
```

`reference: "REFUSE"` marks a question the corpus can't answer - the assistant should decline it.
`expected_sources` are paths relative to `corpus/` (optional).

`uv run eval` syncs this file to a LangSmith dataset, runs every question through the pipeline, and scores:

| Metric | Meaning | Method |
|---|---|---|
| correctness | answer matches the reference | LLM judge (`RAG_JUDGE_MODEL`) |
| groundedness | every claim is supported by retrieved chunks | LLM judge |
| retrieval_hit | expected sources were in the top-k | exact match |
| refusal | declines off-corpus questions, answers in-corpus ones | phrase match |

Each experiment records the system prompt (text + hash), models, chunk size and top-k as metadata, so
you can compare runs side by side in LangSmith (Datasets → `rag-playground-eval` → Experiments).

With `LANGSMITH_TRACING=true`, every question asked in the UI is also traced (retrieval, chunks,
prompt, LLM call) to the `LANGSMITH_PROJECT` project.

## Configuration

All settings live in `.env` (see `.env.example`): models, chunk size/overlap, top-k, history length,
system prompt file, refusal phrase, LangSmith project/dataset names.

`corpus/` is gitignored, so your documents never leave your machine via git. The sample documents
there describe a fictional company - delete them once you add your own
(and replace the questions in `evals/dataset.jsonl`).
