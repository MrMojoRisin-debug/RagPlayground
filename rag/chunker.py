"""Recursive text splitter: prefers headings, then paragraphs, lines, sentences, words."""

import re

from . import config

SEPARATORS = ["\n# ", "\n## ", "\n### ", "\n\n", "\n", ". ", " "]


def _split(text: str, size: int, separators: list[str]) -> list[str]:
    """Split text into pieces no longer than `size`, using the coarsest separator that works."""
    if len(text) <= size:
        return [text]
    for i, sep in enumerate(separators):
        if sep not in text:
            continue
        # Keep the separator attached to the start of the following piece.
        parts = text.split(sep)
        parts = [parts[0]] + [sep.lstrip("\n") + p if sep.startswith("\n") else p for p in parts[1:]]
        joiner = sep if sep in ("\n\n", "\n") or not sep.startswith("\n") else "\n"
        pieces: list[str] = []
        current = ""
        for part in parts:
            candidate = f"{current}{joiner}{part}" if current else part
            if len(candidate) <= size:
                current = candidate
                continue
            if current:
                pieces.append(current)
            if len(part) > size:
                pieces.extend(_split(part, size, separators[i + 1 :]))
                current = ""
            else:
                current = part
        if current:
            pieces.append(current)
        return pieces
    # No separator left: hard cut.
    return [text[i : i + size] for i in range(0, len(text), size)]


def chunk_text(text: str, size: int = config.CHUNK_SIZE, overlap: int = config.CHUNK_OVERLAP) -> list[str]:
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []
    pieces = _split(text, size - overlap, SEPARATORS)
    if overlap <= 0 or len(pieces) == 1:
        return [p.strip() for p in pieces if p.strip()]
    # Prepend the tail of the previous piece so context carries across boundaries.
    chunks = [pieces[0]]
    for prev, cur in zip(pieces, pieces[1:]):
        tail = prev[-overlap:]
        cut = tail.find(" ")
        tail = tail[cut + 1 :] if 0 <= cut < len(tail) - 1 else tail
        chunks.append(f"{tail} {cur}")
    return [c.strip() for c in chunks if c.strip()]


def pack_pages(segments: list[tuple[str, dict]], size: int = config.CHUNK_SIZE - config.CHUNK_OVERLAP) -> list[tuple[str, dict]]:
    """Merge consecutive short PDF pages into windows of up to `size` characters.

    Non-paged segments pass through unchanged. Each window records its page range so
    citations can point at "pp. 12-13". Pages longer than `size` stay on their own and
    are split later by chunk_text.
    """
    if not any("page" in meta for _, meta in segments):
        return segments
    packed: list[tuple[str, dict]] = []
    texts: list[str] = []
    first = last = None
    for text, meta in segments:
        text = text.strip()
        if not text:
            continue
        if texts and sum(map(len, texts)) + len(text) > size:
            packed.append(("\n\n".join(texts), {"page": first, "page_end": last}))
            texts = []
        if not texts:
            first = meta["page"]
        texts.append(text)
        last = meta["page"]
    if texts:
        packed.append(("\n\n".join(texts), {"page": first, "page_end": last}))
    return packed
