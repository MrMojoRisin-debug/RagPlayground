"""Run the RAG pipeline over evals/dataset.jsonl as a LangSmith experiment.

    uv run eval                                   # current settings, default system prompt
    uv run eval --prompt prompts/concise.md       # try a different system prompt
    uv run eval --name shorter-style -c 2         # custom experiment name, lower concurrency

The local dataset.jsonl is the source of truth: each run syncs it to the LangSmith
dataset (adds new questions, updates changed ones, deletes removed ones).
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

from rag import config, embeddings, store
from rag.answer import answer, load_system_prompt

from . import evaluators, report


def _load_local_dataset(path: Path) -> list[dict]:
    rows = []
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if "question" not in row or "reference" not in row:
            sys.exit(f"{path}:{n}: each line needs 'question' and 'reference'")
        rows.append(row)
    return rows


def sync_dataset(client, rows: list[dict]) -> None:
    name = config.LANGSMITH_DATASET
    if client.has_dataset(dataset_name=name):
        dataset = client.read_dataset(dataset_name=name)
    else:
        dataset = client.create_dataset(name, description="RagPlayground eval set (synced from evals/dataset.jsonl)")

    def outputs(row: dict) -> dict:
        default = "off_corpus" if row["reference"] == evaluators.REFUSE else "answerable"
        return {
            "reference": row["reference"],
            "expected_sources": row.get("expected_sources", []),
            "category": row.get("category", default),
        }

    existing = {ex.inputs["question"]: ex for ex in client.list_examples(dataset_id=dataset.id)}
    local = {row["question"]: row for row in rows}

    new = [{"inputs": {"question": q}, "outputs": outputs(r)} for q, r in local.items() if q not in existing]
    if new:
        client.create_examples(dataset_id=dataset.id, examples=new)
    changed = 0
    for q, ex in existing.items():
        if q not in local:
            client.delete_example(ex.id)
        elif ex.outputs != outputs(local[q]):
            client.update_example(ex.id, outputs=outputs(local[q]))
            changed += 1
    removed = len(set(existing) - set(local))
    print(f"Dataset '{name}': {len(local)} examples (+{len(new)} new, ~{changed} updated, -{removed} removed)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a LangSmith eval of the RAG pipeline.")
    parser.add_argument("--prompt", default=None, help="system prompt file (default: RAG_SYSTEM_PROMPT)")
    parser.add_argument("--name", default=None, help="experiment name prefix (default: prompt file name)")
    parser.add_argument("--reranker", choices=["none", "jev"], default=None, help="override RAG_RERANKER")
    parser.add_argument("-c", "--concurrency", type=int, default=4, help="parallel examples (default 4)")
    args = parser.parse_args()

    if args.reranker:
        config.RERANKER = args.reranker
    required = ["LANGSMITH_API_KEY", "OPENROUTER_API_KEY"] + (["TYPESAFE_API_KEY"] if config.RERANKER == "jev" else [])
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        sys.exit(f"Missing {', '.join(missing)} - add to .env (see .env.example).")
    if store.collection().count() == 0:
        sys.exit("The index is empty. Run `uv run ingest` first.")

    from langsmith import Client, evaluate

    prompt_path = Path(args.prompt) if args.prompt else config.SYSTEM_PROMPT_PATH
    prompt_text = load_system_prompt(prompt_path)
    client = Client()
    sync_dataset(client, _load_local_dataset(config.EVAL_DATASET_PATH))
    embeddings.warm_up()  # load once before parallel workers start

    def target(inputs: dict) -> dict:
        result = answer(inputs["question"], prompt_path=prompt_path)
        chunks = result["chunks"]
        return {
            "answer": result["answer"],
            "sources": list(dict.fromkeys(c.source for c in chunks)),  # unique, best match first
            "contexts": [c.text for c in chunks],
        }

    settings = {
        "system_prompt_file": prompt_path.name,
        "system_prompt_sha": hashlib.sha256(prompt_text.encode()).hexdigest()[:10],
        "llm_model": config.LLM_MODEL,
        "judge_model": config.JUDGE_MODEL,
        "embed_model": config.EMBED_MODEL,
        "chunk_size": config.CHUNK_SIZE,
        "chunk_overlap": config.CHUNK_OVERLAP,
        "top_k": config.TOP_K,
        "reranker": config.RERANKER,
        "rerank_candidates": config.RERANK_CANDIDATES if config.RERANKER == "jev" else None,
        "jev_model": config.JEV_MODEL if config.RERANKER == "jev" else None,
        "temperature": config.TEMPERATURE,
    }
    results = evaluate(
        target,
        data=config.LANGSMITH_DATASET,
        evaluators=evaluators.ALL,
        experiment_prefix=args.name or f"{prompt_path.stem}-{config.RERANKER}",
        max_concurrency=args.concurrency,
        client=client,
        metadata={**settings, "system_prompt": prompt_text},
    )
    card = report.build(results, settings)
    path = report.publish(card, client)
    print("\n" + report.render(card))
    print(f"Report saved to {path.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
