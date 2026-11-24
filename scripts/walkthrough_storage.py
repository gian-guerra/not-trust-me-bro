"""Watch one PDF go through ingestion step by step: pages → chunks → vectors → Weaviate.

Usage:
    uv run python scripts/walkthrough_storage.py data/nihms788078.pdf

It deletes that document from Weaviate first, so you see it stored from scratch.
"""

import argparse
import time
from pathlib import Path

from not_trust_me_bro.chunking import chunk_pages
from not_trust_me_bro.config import get_settings
from not_trust_me_bro.embeddings import Embedder
from not_trust_me_bro.ingestion import load_pdf
from not_trust_me_bro.pipeline import fingerprint, ingest_file
from not_trust_me_bro.store import ChunkStore, collection_name, connect


def step(n: int, title: str) -> None:
    print(f"\n{'─' * 70}\nSTEP {n} · {title}\n{'─' * 70}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("pdf", type=Path)
    pdf = p.parse_args().pdf
    s = get_settings()

    step(1, "PDF → pages (ingestion.load_pdf)")
    pages = load_pdf(pdf)
    print(f"{len(pages)} pages kept (references dropped). Page {pages[0].page_number} starts:")
    print(f"  {pages[0].text[:200]!r}")

    step(2, "pages → chunks (chunking.chunk_pages)")
    chunks = chunk_pages(pages, s.chunk_max_tokens, s.chunk_overlap_tokens, s.chunk_min_tokens)
    c = chunks[len(chunks) // 2]
    print(f"{len(chunks)} chunks. A sample one (#{c.chunk_index}):")
    print(f"  pages {c.page_start}-{c.page_end} · section {c.section!r}")
    print(f"  {c.text[:200]!r} ...")

    step(3, f"chunks → vectors (embeddings.Embedder, {s.embedding_model})")
    embedder = Embedder(s.embedding_model, s.embedding_query_prefix)
    t = time.time()
    vectors = embedder.embed_documents([ch.text for ch in chunks])
    print(f"{len(vectors)} vectors × {len(vectors[0])} numbers in {time.time() - t:.1f}s")
    print(f"  chunk #{c.chunk_index} → {[round(x, 3) for x in vectors[c.chunk_index][:6]]} ...")
    print("  one vector per chunk: the whole passage's meaning compressed into 384 numbers")

    with connect(s) as client:
        store = ChunkStore(client, collection_name(s.embedding_model))

        step(4, f"vectors + text + metadata → Weaviate collection {store.name!r}")
        store.delete_document(pdf.name)
        print(f"objects for {pdf.name} before: {store.count(pdf.name)}")
        store.replace_document(chunks, vectors, fingerprint(pdf, s))
        print(f"objects for {pdf.name} after:  {store.count(pdf.name)}")
        print(f"objects in the whole collection: {store.count()}")
        print(
            "Weaviate updated its BM25 index (text), HNSW index (vectors) and filters (metadata)."
        )

        step(5, "ingest the same file again (pipeline.ingest_file)")
        result = ingest_file(pdf, store, embedder, s)
        print(f"status: {result.status!r} → fingerprint unchanged, nothing re-embedded")

    step(6, "look inside Weaviate yourself (open in a browser)")
    base = f"http://{s.weaviate_host}:{s.weaviate_http_port}/v1"
    name = collection_name(s.embedding_model)
    print(f"  schema (fields and index settings): {base}/schema/{name}")
    print(f"  one stored object:                  {base}/objects?class={name}&limit=1")
    print(
        f"  the same, with its vector:          {base}/objects?class={name}&limit=1&include=vector"
    )


if __name__ == "__main__":
    main()
