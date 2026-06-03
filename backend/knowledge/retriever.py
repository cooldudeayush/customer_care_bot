"""Document retrieval — the grounded-knowledge layer (Phase 2).

A lightweight, dependency-light vector retriever: it chunks the markdown policy
corpus, embeds each chunk with Gemini embeddings, and answers queries by cosine
similarity. The index persists to a JSON file so we embed the corpus ONCE and
reuse it across runs (the free-tier-friendly approach the architecture calls for).

This sits behind a small ``Retriever`` protocol so a heavier implementation
(LightRAG's entity+vector graph) can drop in later without touching the agent
loop — only the implementation here changes.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from config import get_settings
from llm import LLMNotConfigured, gemini_client

logger = logging.getLogger("ccb.knowledge")


@dataclass
class RetrievedChunk:
    text: str
    source: str
    heading: str
    score: float = 0.0


class Retriever(Protocol):
    async def search(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]: ...
    @property
    def size(self) -> int: ...


# ---------------------------------------------------------------------------
# Markdown chunking
# ---------------------------------------------------------------------------
@dataclass
class _Chunk:
    text: str
    source: str
    heading: str


def chunk_markdown(text: str, source: str) -> list[_Chunk]:
    """Split a markdown doc into one chunk per ``## H2`` section.

    Each chunk is prefixed with the doc's H1 title + the section heading so the
    embedding (and the grounding context shown to the LLM) carries its context.
    """
    title = ""
    chunks: list[_Chunk] = []
    heading: str | None = None
    body: list[str] = []

    def flush() -> None:
        if heading is not None:
            content = "\n".join(body).strip()
            if content:
                label = f"{title} — {heading}" if title else heading
                chunks.append(_Chunk(text=f"{label}\n{content}", source=source, heading=heading))

    for line in text.splitlines():
        if line.startswith("## "):
            flush()
            heading = line[3:].strip()
            body = []
        elif line.startswith("# "):
            title = line[2:].strip()
        elif heading is not None:
            body.append(line)
    flush()

    # Files with no H2 sections: treat the whole doc as one chunk.
    if not chunks and text.strip():
        chunks.append(_Chunk(text=text.strip(), source=source, heading=title or source))
    return chunks


def load_corpus_chunks(corpus_dir: str) -> list[_Chunk]:
    chunks: list[_Chunk] = []
    if not os.path.isdir(corpus_dir):
        logger.warning("Corpus dir not found: %s", os.path.abspath(corpus_dir))
        return chunks
    for name in sorted(os.listdir(corpus_dir)):
        if not name.endswith((".md", ".txt")):
            continue
        path = os.path.join(corpus_dir, name)
        with open(path, "r", encoding="utf-8") as f:
            chunks.extend(chunk_markdown(f.read(), source=name))
    return chunks


# ---------------------------------------------------------------------------
# Vector retriever
# ---------------------------------------------------------------------------
def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class VectorRetriever:
    """Cosine-similarity retriever over Gemini-embedded corpus chunks."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._meta: list[dict] = []          # parallel to _matrix rows
        self._matrix: np.ndarray | None = None  # (N, dim), L2-normalized

    @property
    def size(self) -> int:
        return len(self._meta)

    # -- build / persist ----------------------------------------------------
    async def build(self) -> int:
        """Embed the corpus and build the in-memory index. Returns chunk count.

        Raises LLMNotConfigured if there is no API key (caller decides what to do).
        """
        chunks = load_corpus_chunks(self._settings.corpus_path)
        if not chunks:
            logger.warning("No corpus chunks to index.")
            self._meta, self._matrix = [], None
            return 0
        vectors = await gemini_client.embed_texts(
            [c.text for c in chunks], task_type="RETRIEVAL_DOCUMENT"
        )
        matrix = _normalize(np.array(vectors, dtype=np.float32))
        self._meta = [
            {"text": c.text, "source": c.source, "heading": c.heading}
            for c in chunks
        ]
        self._matrix = matrix
        logger.info("Built retrieval index: %d chunks, dim=%d", len(chunks), matrix.shape[1])
        return len(chunks)

    def save(self) -> None:
        if self._matrix is None:
            return
        path = self._settings.index_path
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        payload = {
            "model": self._settings.gemini_embed_model,
            "dim": int(self._matrix.shape[1]),
            "items": [
                {**m, "vector": self._matrix[i].tolist()}
                for i, m in enumerate(self._meta)
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        logger.info("Saved retrieval index -> %s", os.path.abspath(path))

    def load(self) -> bool:
        """Load a persisted index if present. Returns True on success."""
        path = self._settings.index_path
        if not os.path.isfile(path):
            return False
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            required = ("text", "source", "heading", "vector")
            items = [
                it
                for it in payload.get("items", [])
                if isinstance(it, dict) and all(k in it for k in required)
            ]
            if not items:
                logger.warning("Retrieval index has no usable items: %s", path)
                return False
            self._meta = [
                {"text": it["text"], "source": it["source"], "heading": it["heading"]}
                for it in items
            ]
            self._matrix = _normalize(
                np.array([it["vector"] for it in items], dtype=np.float32)
            )
            logger.info("Loaded retrieval index: %d chunks", len(self._meta))
            return True
        except Exception:  # noqa: BLE001 - a bad index file shouldn't crash startup
            logger.exception("Failed to load retrieval index")
            return False

    # -- query --------------------------------------------------------------
    async def search(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        if self._matrix is None or not self._meta:
            return []
        k = top_k or self._settings.retrieval_top_k
        try:
            qvec = await gemini_client.embed_query(query)
        except LLMNotConfigured:
            return []
        if not qvec:
            return []
        q = np.array(qvec, dtype=np.float32)
        qn = np.linalg.norm(q)
        # Guard against a zero or non-finite (NaN/inf) query vector — np.nan == 0
        # is False, so a bare `qn == 0` check would let NaN through and poison sims.
        if not np.isfinite(qn) or qn == 0:
            return []
        q = q / qn
        if not np.all(np.isfinite(q)):
            return []
        sims = self._matrix @ q
        top = np.argsort(-sims)[: max(1, k)]
        return [
            RetrievedChunk(
                text=self._meta[i]["text"],
                source=self._meta[i]["source"],
                heading=self._meta[i]["heading"],
                score=float(sims[i]),
            )
            for i in top
        ]


# Process-wide retriever instance used by the agent loop.
retriever = VectorRetriever()
