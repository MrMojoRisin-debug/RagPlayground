"""Scoring functions for LangSmith experiments.

Each evaluator receives the example's inputs, the pipeline's outputs, and the
reference outputs, and returns {"key", "score", "comment"}. A score of None means
"not applicable to this example" (e.g. correctness on a question that should be refused).
"""

import json
import re

from rag import config
from rag.answer import llm_client

REFUSE = "REFUSE"


def _norm(text: str) -> str:
    return text.lower().replace("’", "'").strip()


def is_refusal(answer: str) -> bool:
    return _norm(config.REFUSAL_TEXT) in _norm(answer)


def _judge(prompt: str) -> tuple[float | None, str]:
    resp = llm_client().chat.completions.create(
        model=config.JUDGE_MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "You are a strict evaluator. Respond with a single JSON object: "
                '{"reasoning": "<one or two sentences>", "score": <number between 0 and 1>}',
            },
            {"role": "user", "content": prompt},
        ],
    )
    text = resp.choices[0].message.content or ""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        data = json.loads(match.group(0)) if match else {}
        return max(0.0, min(1.0, float(data["score"]))), str(data.get("reasoning", ""))
    except (ValueError, KeyError, TypeError):
        return None, f"Could not parse judge output: {text[:200]}"


def refusal(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """Off-corpus questions must be declined; in-corpus questions must be answered."""
    should_refuse = reference_outputs.get("reference") == REFUSE
    refused = is_refusal(outputs["answer"])
    if should_refuse:
        comment = "correctly refused" if refused else "answered a question the corpus doesn't cover"
    else:
        comment = "wrongly refused" if refused else "answered"
    return {"key": "refusal", "score": int(refused == should_refuse), "comment": comment}


def retrieval_hit(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """Fraction of expected_sources that appeared in the retrieved chunks."""
    expected = reference_outputs.get("expected_sources") or []
    if not expected:
        return {"key": "retrieval_hit", "score": None, "comment": "no expected_sources"}
    retrieved = set(outputs.get("sources", []))
    found = [s for s in expected if s in retrieved]
    missing = sorted(set(expected) - set(found))
    return {
        "key": "retrieval_hit",
        "score": len(found) / len(expected),
        "comment": f"missing: {missing}" if missing else "all expected sources retrieved",
    }


def correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """LLM judge: does the answer agree with the reference answer?"""
    reference = reference_outputs.get("reference")
    if reference == REFUSE:
        return {"key": "correctness", "score": None, "comment": "refusal case (see refusal score)"}
    score, reasoning = _judge(
        f"Question: {inputs['question']}\n\nReference answer: {reference}\n\n"
        f"Candidate answer: {outputs['answer']}\n\n"
        "Score 1 if the candidate conveys the same key facts as the reference with no contradictions, "
        "0.5 if it is partially correct or misses an important fact, 0 if it is wrong, contradicts the "
        "reference, or declines to answer. Extra correct detail is fine. Ignore citation markers like [1]."
    )
    return {"key": "correctness", "score": score, "comment": reasoning}


def groundedness(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """LLM judge: is every claim in the answer supported by the retrieved chunks?"""
    if is_refusal(outputs["answer"]):
        return {"key": "groundedness", "score": None, "comment": "refusal (nothing to ground)"}
    contexts = "\n\n".join(f"[{i}] {c}" for i, c in enumerate(outputs.get("contexts", []), 1))
    score, reasoning = _judge(
        f"Retrieved sources:\n{contexts or '(none)'}\n\nAnswer: {outputs['answer']}\n\n"
        "Score 1 if every factual claim in the answer is directly supported by the sources, "
        "0.5 if minor claims are unsupported, 0 if any key claim is unsupported or fabricated."
    )
    return {"key": "groundedness", "score": score, "comment": reasoning}


ALL = [correctness, groundedness, retrieval_hit, refusal]
