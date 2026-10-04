"""ChromaDB collection + ingest manifest (what was indexed, and with which settings)."""

import json
from functools import lru_cache

from . import config


@lru_cache(maxsize=1)
def _client():
    import chromadb

    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(config.CHROMA_DIR))


def collection():
    return _client().get_or_create_collection(config.COLLECTION_NAME, metadata={"hnsw:space": "cosine"})


def reset_collection():
    client = _client()
    try:
        client.delete_collection(config.COLLECTION_NAME)
    except Exception:
        pass
    return collection()


def index_settings() -> dict:
    """Settings that, if changed, invalidate every stored embedding."""
    return {
        "embed_model": config.EMBED_MODEL,
        "chunk_size": config.CHUNK_SIZE,
        "chunk_overlap": config.CHUNK_OVERLAP,
    }


def load_manifest() -> dict:
    if config.MANIFEST_PATH.exists():
        return json.loads(config.MANIFEST_PATH.read_text())
    return {"settings": None, "files": {}, "last_sync": None}


def save_manifest(manifest: dict) -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
