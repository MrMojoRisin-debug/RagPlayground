"""Turn LangSmith experiment results into a scorecard.

Prints it, saves it as evals/reports/<experiment>.md, and logs `overall` and `pass_rate` to the
experiment as summary feedback so they appear as columns in LangSmith's Experiments table.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .evaluators import JUDGE_ERROR, METRICS, REFUSE, is_refusal

REPORTS_DIR = Path(__file__).resolve().parent / "reports"

LABELS = {
    "groundedness": ("Groundedness", "every claim in the answer is supported by the retrieved chunks"),
    "correctness": ("Correctness", "answer matches the reference answer"),
    "retrieval": ("Retrieval", "expected source document appears in the top-k chunks"),
    "guardrail": ("Guardrails", "declines off-corpus, harmful, injection and prompt-leak questions"),
}


@dataclass
class Scorecard:
    experiment: str
    url: str | None
    settings: dict
    metrics: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    guardrail_by_category: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    failures: list[tuple[str, str, float, str]] = field(default_factory=list)  # metric, question, score, comment
    over_refusals: list[str] = field(default_factory=list)
    unscored: list[tuple[str, str, str]] = field(default_factory=list)  # metric, question, comment
    ranks: list[float] = field(default_factory=list)  # reciprocal rank per question
    passed: int = 0
    total: int = 0

    def mean(self, key: str) -> float | None:
        vals = self.metrics.get(key) or []
        return sum(vals) / len(vals) if vals else None

    @property
    def overall(self) -> float | None:
        means = [m for k in METRICS if (m := self.mean(k)) is not None]
        return sum(means) / len(means) if means else None

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0


def build(results, settings: dict) -> Scorecard:
    card = Scorecard(experiment=results.experiment_name, url=None, settings=settings)
    for row in results:
        question = row["example"].inputs["question"]
        reference = row["example"].outputs or {}
        answer = (row["run"].outputs or {}).get("answer", "")
        card.total += 1
        perfect = True
        for res in row["evaluation_results"]["results"]:
            if res.key == "retrieval_rank" and res.score is not None:
                card.ranks.append(float(res.score))
                continue
            if res.key not in METRICS:
                continue
            if res.score is None:
                if (res.comment or "").startswith(JUDGE_ERROR):
                    card.unscored.append((res.key, question, res.comment))
                    perfect = False
                continue
            score = float(res.score)
            card.metrics[res.key].append(score)
            if res.key == "guardrail":
                card.guardrail_by_category[reference.get("category", "off_corpus")].append(score)
            if score < 1:
                perfect = False
                card.failures.append((res.key, question, score, res.comment or ""))
        if reference.get("reference") != REFUSE and is_refusal(answer):
            card.over_refusals.append(question)
        card.passed += perfect
    return card


def _bar(score: float, width: int = 20) -> str:
    filled = round(score * width)
    return "█" * filled + "░" * (width - filled)


def render(card: Scorecard) -> str:
    s = card.settings
    lines = [
        f"# RAG eval report: {card.experiment}",
        "",
        f"- Date: {datetime.now():%Y-%m-%d %H:%M}",
        f"- System prompt: `{s['system_prompt_file']}` (sha {s['system_prompt_sha']})",
        f"- LLM: `{s['llm_model']}` · judge: `{s['judge_model']}`",
        f"- Retrieval: `{s['embed_model']}`, top-k {s['top_k']}, chunk {s['chunk_size']}/{s['chunk_overlap']} chars",
        f"- Reranker: " + (f"Jev `{s['jev_model']}` over top {s['rerank_candidates']}" if s.get("reranker") == "jev" else "none"),
    ]
    if card.url:
        lines.append(f"- LangSmith: {card.url}")
    lines += ["", "```", f"{'Metric':<14}{'Score':>7}  {'':20}  {'n':>3}   What it measures", "-" * 100]
    for key in METRICS:
        label, desc = LABELS[key]
        m = card.mean(key)
        n = len(card.metrics.get(key, []))
        lines.append(
            f"{label:<14}{m:>7.2f}  {_bar(m)}  {n:>3}   {desc}" if m is not None else f"{label:<14}{'n/a':>7}  {'':20}  {n:>3}   {desc}"
        )
    lines.append("-" * 100)
    if card.overall is not None:
        lines.append(f"{'OVERALL':<14}{card.overall:>7.2f}  {_bar(card.overall)}       mean of the four metrics")
    lines.append(
        f"{'Pass rate':<14}{card.pass_rate:>7.2f}  {_bar(card.pass_rate)}  {card.total:>3}   "
        f"{card.passed}/{card.total} questions perfect on every applicable metric"
    )
    if card.ranks:
        mrr = sum(card.ranks) / len(card.ranks)
        lines.append(f"{'Retrieval MRR':<14}{mrr:>7.2f}  {_bar(mrr)}  {len(card.ranks):>3}   "
                     "how high the right document ranks (1 = always first; not part of overall)")
    lines.append("```")

    if card.guardrail_by_category:
        lines += ["", "## Guardrails by category", "", "| Category | Score | n |", "|---|---|---|"]
        for cat, vals in sorted(card.guardrail_by_category.items()):
            lines.append(f"| {cat} | {sum(vals) / len(vals):.2f} | {len(vals)} |")

    if card.unscored:
        lines += ["", f"## Unscored - judge failed ({len(card.unscored)})", "",
                  "Not counted in the averages above; check these by hand in LangSmith.", ""]
        lines += [f"- {LABELS[k][0]}: {q}  \n  {c}" for k, q, c in card.unscored]

    if card.over_refusals:
        lines += ["", f"## Over-refusals ({len(card.over_refusals)})", "", "Answerable questions the assistant declined:", ""]
        lines += [f"- {q}" for q in card.over_refusals]

    lines += ["", f"## Below-perfect results ({len(card.failures)})", ""]
    if not card.failures:
        lines.append("None.")
    for key, question, score, comment in sorted(card.failures, key=lambda f: (METRICS.index(f[0]), f[2])):
        lines.append(f"- **{LABELS[key][0]} {score:.2f}**: {question}  \n  {comment}")
    return "\n".join(lines) + "\n"


def publish(card: Scorecard, client) -> Path:
    """Log summary scores to the LangSmith experiment and save the markdown report."""
    project = client.read_project(project_name=card.experiment)
    card.url = getattr(project, "url", None)
    for key, score, comment in [
        ("overall", card.overall, "mean of groundedness, correctness, retrieval, guardrail"),
        ("pass_rate", card.pass_rate, f"{card.passed}/{card.total} questions perfect on every metric"),
        ("retrieval_mrr", sum(card.ranks) / len(card.ranks) if card.ranks else None, "mean reciprocal rank of the expected source"),
    ]:
        if score is not None:
            client.create_feedback(None, key=key, score=score, comment=comment, project_id=project.id)
    REPORTS_DIR.mkdir(exist_ok=True)
    path = REPORTS_DIR / f"{card.experiment}.md"
    path.write_text(render(card))
    return path
