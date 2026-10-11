"""Rerank retrieved chunks with Jev.

Exact search returns the top RAG_RERANK_CANDIDATES chunks by embedding similarity. Jev rates
each one against the question on a 0-3 rubric in a single request, and the best RAG_TOP_K go
to the LLM. If Jev fails, the embedding order is kept, so a question never fails because of
the reranker.
"""

import logging

from langsmith import traceable

from . import config, jev

log = logging.getLogger(__name__)

RUBRIC = [
    "Unrelated to the question",
    "Same topic, but does not help answer the question",
    "Partially answers the question, or gives context needed to answer it",
    "Directly answers the question",
]


@traceable(run_type="chain", name="rerank_jev")
def rerank(question: str, chunks: list, k: int) -> list:
    """Reorder `chunks` (best embedding match first) by Jev relevance; return the top k."""
    if len(chunks) <= 1:
        return chunks[:k]
    state = {
        "question": question,
        "passages": [{"source": c.label, "text": c.text} for c in chunks],
    }
    questions = {
        f"p{i}": jev.score(
            f"How useful is `passages[{i}].text` for answering `question`?", RUBRIC
        )
        for i in range(len(chunks))
    }
    try:
        answers = jev.system_one(state, questions)["answers"]
        for i, c in enumerate(chunks):
            c.rerank_score = float(answers[f"p{i}"]["score"])
    except (jev.JevError, KeyError, TypeError, ValueError) as exc:
        log.warning("Jev rerank failed, keeping embedding order: %s", exc)
        return chunks[:k]
    # Jev's expected level first; embedding similarity breaks ties.
    ranked = sorted(chunks, key=lambda c: (c.rerank_score, c.score), reverse=True)
    return ranked[:k]


def candidates_for(k: int) -> int:
    """How many chunks to fetch before reranking."""
    return max(k, config.RERANK_CANDIDATES) if config.RERANKER == "jev" else k
