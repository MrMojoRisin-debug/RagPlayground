"""Find the chunks most relevant to a question.

Uses exact (brute-force) cosine search over all chunk embeddings instead of Chroma's HNSW
index. HNSW is approximate: small, distinctive documents ingested next to a large corpus can
end up in a graph region that greedy search never reaches, so they silently disappear from
results. At this scale (tens of thousands of chunks) exact search is a single matrix
multiply that takes milliseconds, and is always correct. Chroma remains the storage layer.
"""

import threading
from dataclasses import dataclass

import numpy as np
from langsmith import traceable

from . import config, embeddings, rerank, store


@dataclass
class Chunk:
    text: str
    source: str
    page: int | None
    page_end: int | None
    score: float  # cosine similarity, higher is better
    rerank_score: float | None = None  # Jev relevance, 0 (unrelated) to 3 (directly answers)

    @property
    def label(self) -> str:
        if not self.page:
            return self.source
        if self.page_end and self.page_end != self.page:
            return f"{self.source} (pp. {self.page}-{self.page_end})"
        return f"{self.source} (p. {self.page})"


_lock = threading.Lock()
_cache: dict = {"version": None, "ids": [], "metas": [], "matrix": None}


def _index():
    """All chunk ids, metadata and embeddings, reloaded from Chroma only after an ingest."""
    version = (store.load_manifest().get("last_sync"), store.collection().count())
    with _lock:
        if _cache["version"] != version:
            col = store.collection()
            ids, metas, vectors = [], [], []
            batch = 5000
            for offset in range(0, col.count(), batch):
                got = col.get(limit=batch, offset=offset, include=["embeddings", "metadatas"])
                ids += got["ids"]
                metas += got["metadatas"]
                vectors.append(np.asarray(got["embeddings"], dtype=np.float32))
            matrix = np.vstack(vectors) if vectors else np.zeros((0, 0), dtype=np.float32)
            _cache.update(version=version, ids=ids, metas=metas, matrix=matrix)
        return _cache["ids"], _cache["metas"], _cache["matrix"]


@traceable(run_type="retriever", name="retrieve")
def retrieve(question: str, k: int | None = None) -> list[Chunk]:
    """Top-k chunks for the question; with RAG_RERANKER=jev, reranked from a wider candidate set."""
    k = k or config.TOP_K
    chunks = search(question, rerank.candidates_for(k))
    if config.RERANKER == "jev":
        return rerank.rerank(question, chunks, k)
    return chunks[:k]


@traceable(name="search")
def search(question: str, k: int) -> list[Chunk]:
    """Exact cosine search: the k most similar chunks, best first."""
    ids, metas, matrix = _index()
    if not ids:
        return []
    query = np.asarray(embeddings.embed_query(question), dtype=np.float32)
    scores = matrix @ query  # vectors are normalised, so the dot product is cosine similarity
    k = min(k, len(ids))
    top = np.argpartition(-scores, k - 1)[:k]
    top = top[np.argsort(-scores[top])]
    got = store.collection().get(ids=[ids[i] for i in top], include=["documents"])
    docs = dict(zip(got["ids"], got["documents"]))
    return [
        Chunk(
            text=docs[ids[i]],
            source=metas[i]["source"],
            page=metas[i].get("page"),
            page_end=metas[i].get("page_end"),
            score=float(scores[i]),
        )
        for i in top
    ]
