"""Local embedding model (sentence-transformers). Loaded once per process."""

import threading

from . import config

# One lock guards both loading and encoding: the GPU (MPS) backend crashes if two threads
# initialise or run the model at the same time (e.g. parallel eval workers).
_lock = threading.RLock()
_loaded = None


def _model():
    global _loaded
    with _lock:
        if _loaded is None:
            _loaded = _load()
        return _loaded


def _load():
    import torch
    from sentence_transformers import SentenceTransformer

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(config.EMBED_MODEL, device=device)
    # Chunks are ~800 tokens; capping the window keeps memory use predictable.
    model.max_seq_length = min(model.max_seq_length or 2048, 2048)
    return model


def _encode(texts: list[str], prompt_name: str, batch_size: int) -> list[list[float]]:
    model = _model()
    # Instruction-aware models (Qwen3, nomic, e5...) ship named prompts for queries/documents.
    kwargs = {"prompt_name": prompt_name} if prompt_name in (model.prompts or {}) else {}
    with _lock:  # the model isn't safe to call from several threads at once
        vectors = model.encode(
            texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False, **kwargs
        )
    return vectors.tolist()


def embed_documents(texts: list[str]) -> list[list[float]]:
    return _encode(texts, "document", batch_size=8)


def embed_query(text: str) -> list[float]:
    return _encode([text], "query", batch_size=1)[0]


def warm_up() -> None:
    _model()
