"""Ingestion pipeline: PDF → pages → chunks → vectors → Weaviate.

Re-ingesting is idempotent. A document's fingerprint hashes its bytes *and* the chunking
settings: if neither changed, the file is skipped (no re-embedding); if either changed, all
of its old chunks are replaced. Replacing, not upserting by id, matters: a new chunk size can
produce fewer chunks, and the leftover old ones would otherwise stay as stale duplicates.
"""

import hashlib
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from not_trust_me_bro.chunking import chunk_pages
from not_trust_me_bro.config import Settings
from not_trust_me_bro.ingestion import load_pdf
from not_trust_me_bro.store import ChunkStore

# One INFO line per stage, with its duration. The CLI prints them under the document's name.
log = logging.getLogger(__name__)


class DocumentEmbedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...


@dataclass(frozen=True)
class IngestResult:
    source: str
    status: Literal["ingested", "unchanged"]
    chunks: int


def fingerprint(path: Path, settings: Settings) -> str:
    h = hashlib.sha256(path.read_bytes())
    config = (settings.chunk_max_tokens, settings.chunk_overlap_tokens, settings.chunk_min_tokens)
    h.update(repr(config).encode())
    return h.hexdigest()


def ingest_file(
    path: Path,
    store: ChunkStore,
    embedder: DocumentEmbedder,
    settings: Settings,
    force: bool = False,
) -> IngestResult:
    fp = fingerprint(path, settings)
    if not force and store.stored_fingerprint(path.name) == fp:
        return IngestResult(path.name, "unchanged", store.count(path.name))

    t = time.perf_counter()
    pages = load_pdf(path)
    log.info("read %d pages (references dropped) · %.1fs", len(pages), time.perf_counter() - t)

    t = time.perf_counter()
    chunks = chunk_pages(
        pages,
        max_tokens=settings.chunk_max_tokens,
        overlap_tokens=settings.chunk_overlap_tokens,
        min_tokens=settings.chunk_min_tokens,
    )
    log.info("split into %d chunks · %.1fs", len(chunks), time.perf_counter() - t)

    t = time.perf_counter()
    vectors = embedder.embed_documents([c.text for c in chunks])
    log.info("embedded %d chunks · %.1fs", len(vectors), time.perf_counter() - t)

    t = time.perf_counter()
    store.replace_document(chunks, vectors, fp)
    log.info("stored in Weaviate · %.1fs", time.perf_counter() - t)
    return IngestResult(path.name, "ingested", len(chunks))
