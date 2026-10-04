"""Find the chunks most relevant to a question."""

from dataclasses import dataclass

from langsmith import traceable

from . import config, embeddings, store


@dataclass
class Chunk:
    text: str
    source: str
    page: int | None
    score: float  # cosine similarity, higher is better

    @property
    def label(self) -> str:
        return f"{self.source} (p. {self.page})" if self.page else self.source


@traceable(run_type="retriever", name="retrieve")
def retrieve(question: str, k: int = config.TOP_K) -> list[Chunk]:
    col = store.collection()
    if col.count() == 0:
        return []
    res = col.query(query_embeddings=[embeddings.embed_query(question)], n_results=min(k, col.count()))
    return [
        Chunk(text=doc, source=meta["source"], page=meta.get("page"), score=1 - dist)
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]
