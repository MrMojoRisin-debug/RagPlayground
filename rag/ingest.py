"""Sync corpus/ into the vector index.

Incremental by default: new and changed files are (re)embedded, deleted files are
removed, unchanged files are skipped. Use --rebuild to start from scratch.

    uv run ingest            # incremental sync
    uv run ingest --rebuild  # wipe and re-index everything
"""

import argparse
import contextlib
import hashlib
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

from . import config, embeddings, store
from .chunker import chunk_text, pack_pages
from .loaders import SUPPORTED_EXTENSIONS, load_file


@contextlib.contextmanager
def _keep_awake():
    """On macOS, block idle sleep while this process runs (a long ingest otherwise stalls when the Mac sleeps)."""
    proc = None
    if sys.platform == "darwin" and shutil.which("caffeinate"):
        proc = subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])
    try:
        yield
    finally:
        if proc:
            proc.terminate()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _scan_corpus(log: Callable[[str], None]) -> dict[str, Path]:
    files = {}
    for path in sorted(config.CORPUS_DIR.rglob("*")):
        if not path.is_file() or any(part.startswith(".") for part in path.relative_to(config.CORPUS_DIR).parts):
            continue
        rel = path.relative_to(config.CORPUS_DIR).as_posix()
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files[rel] = path
        else:
            log(f"  skip  {rel} (unsupported type)")
    return files


BATCH = 256  # chunks embedded + written per step (Chroma caps a single add at ~5k records)


def _index_file(col, rel: str, path: Path, log: Callable[[str], None]) -> int:
    ids, docs, metas = [], [], []
    for text, extra in pack_pages(load_file(path)):
        for chunk in chunk_text(text):
            n = len(ids)
            ids.append(f"{rel}::{n}")
            docs.append(chunk)
            metas.append({"source": rel, "chunk": n, **extra})
    col.delete(where={"source": rel})
    started = time.time()
    for i in range(0, len(ids), BATCH):
        j = i + BATCH
        col.add(ids=ids[i:j], documents=docs[i:j], metadatas=metas[i:j], embeddings=embeddings.embed_documents(docs[i:j]))
        if len(ids) > BATCH:
            done = min(j, len(ids))
            eta = (time.time() - started) / done * (len(ids) - done)
            log(f"        {rel}: {done}/{len(ids)} chunks embedded, ~{eta / 60:.0f} min left")
    return len(ids)


def sync(rebuild: bool = False, log: Callable[[str], None] = print) -> dict:
    with _keep_awake():
        return _sync(rebuild, log)


def _sync(rebuild: bool, log: Callable[[str], None]) -> dict:
    config.CORPUS_DIR.mkdir(exist_ok=True)
    manifest = store.load_manifest()
    settings = store.index_settings()

    if not rebuild and manifest["files"] and manifest["settings"] != settings:
        log("Index settings changed (embedding model or chunking) - rebuilding everything.")
        rebuild = True
    if rebuild:
        col = store.reset_collection()
        manifest = {"settings": settings, "files": {}, "last_sync": None}
    else:
        col = store.collection()
        manifest["settings"] = settings

    on_disk = _scan_corpus(log)
    stats = {"added": 0, "updated": 0, "removed": 0, "unchanged": 0, "failed": 0, "chunks": 0}
    started = time.time()

    for rel in sorted(set(manifest["files"]) - set(on_disk)):
        col.delete(where={"source": rel})
        del manifest["files"][rel]
        stats["removed"] += 1
        log(f"  -     {rel}")

    try:
        for rel, path in on_disk.items():
            digest = _file_hash(path)
            previous = manifest["files"].get(rel)
            if previous and previous["hash"] == digest:
                stats["unchanged"] += 1
                continue
            try:
                n_chunks = _index_file(col, rel, path, log)
            except Exception as exc:  # keep going; the file is retried next sync
                stats["failed"] += 1
                log(f"  !     {rel}: {exc}")
                continue
            manifest["files"][rel] = {"hash": digest, "chunks": n_chunks}
            stats["updated" if previous else "added"] += 1
            stats["chunks"] += n_chunks
            log(f"  {'~' if previous else '+'}     {rel} ({n_chunks} chunks)")
    finally:
        manifest["last_sync"] = datetime.now().isoformat(timespec="seconds")
        store.save_manifest(manifest)

    log(
        f"Done in {time.time() - started:.1f}s: +{stats['added']} added, ~{stats['updated']} updated, "
        f"-{stats['removed']} removed, {stats['unchanged']} unchanged"
        + (f", {stats['failed']} failed" if stats["failed"] else "")
        + f". Index holds {col.count()} chunks from {len(manifest['files'])} files."
    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync corpus/ into the vector index.")
    parser.add_argument("--rebuild", action="store_true", help="wipe the index and re-embed every file")
    args = parser.parse_args()
    print(f"Syncing {config.CORPUS_DIR} (embedding model: {config.EMBED_MODEL})")
    stats = sync(rebuild=args.rebuild, log=lambda line: print(line, flush=True))
    sys.exit(1 if stats["failed"] else 0)


if __name__ == "__main__":
    main()
