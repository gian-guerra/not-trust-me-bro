"""Store and pipeline tests against a real Weaviate (docker compose up -d), in a throwaway
collection. Vectors are fake, so no embedding model is needed. Skipped if Weaviate is down."""

import uuid
from pathlib import Path

import pymupdf
import pytest

from not_trust_me_bro.chunking import Chunk
from not_trust_me_bro.config import Settings
from not_trust_me_bro.pipeline import ingest_file
from not_trust_me_bro.store import ChunkStore, collection_name, connect

DIMS = 4


class FakeEmbedder:
    def __init__(self) -> None:
        self.calls = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [[1.0, 0.0, 0.0, float(i)] for i in range(len(texts))]


@pytest.fixture
def store():
    try:
        client = connect(Settings())
        client.is_ready()
    except Exception:
        pytest.skip("Weaviate is not running (docker compose up -d)")
    name = f"Test_{uuid.uuid4().hex[:8]}"
    yield ChunkStore(client, name)
    client.collections.delete(name)
    client.close()


def chunks(source: str, n: int) -> list[Chunk]:
    return [Chunk(source, i, f"chunk {i} text", i + 1, i + 1, "Intro") for i in range(n)]


def vectors(n: int) -> list[list[float]]:
    return [[1.0, 0.0, 0.0, float(i)] for i in range(n)]


def test_collection_name_is_derived_from_the_model():
    assert collection_name("BAAI/bge-small-en-v1.5") == "Chunks_bge_small_en_v1_5"


def test_replacing_a_document_twice_does_not_duplicate(store: ChunkStore):
    store.replace_document(chunks("a.pdf", 5), vectors(5), "fp1")
    store.replace_document(chunks("a.pdf", 5), vectors(5), "fp1")
    assert store.count("a.pdf") == 5


def test_fewer_new_chunks_leave_no_stale_ones_behind(store: ChunkStore):
    store.replace_document(chunks("a.pdf", 5), vectors(5), "fp1")
    store.replace_document(chunks("a.pdf", 3), vectors(3), "fp2")
    assert store.count("a.pdf") == 3
    assert store.stored_fingerprint("a.pdf") == "fp2"


def test_documents_are_isolated_from_each_other(store: ChunkStore):
    store.replace_document(chunks("a.pdf", 2), vectors(2), "fpa")
    store.replace_document(chunks("b.pdf", 3), vectors(3), "fpb")
    store.delete_document("a.pdf")
    assert (store.count("a.pdf"), store.count("b.pdf"), store.count()) == (0, 3, 3)


def test_incomplete_document_has_no_fingerprint(store: ChunkStore):
    store.replace_document(chunks("a.pdf", 3), vectors(3), "fp1")
    first = store.collection.query.fetch_objects(limit=1).objects[0]
    store.collection.data.delete_by_id(first.uuid)  # simulate a crash mid-insert
    assert store.stored_fingerprint("a.pdf") is None


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    text = ("Status hierarchies form quickly in new groups and shape attention. " * 6 + "\n\n") * 3
    doc = pymupdf.open()
    doc.new_page().insert_textbox(pymupdf.Rect(72, 100, 520, 700), text, fontsize=11)
    path = tmp_path / "sample.pdf"
    doc.save(path)
    return path


def test_ingesting_an_unchanged_file_skips_embedding(store: ChunkStore, sample_pdf: Path):
    embedder, settings = FakeEmbedder(), Settings()

    first = ingest_file(sample_pdf, store, embedder, settings)
    second = ingest_file(sample_pdf, store, embedder, settings)

    assert first.status == "ingested" and first.chunks > 0
    assert second.status == "unchanged"
    assert embedder.calls == 1
    assert store.count("sample.pdf") == first.chunks


def test_changing_chunk_settings_reingests(store: ChunkStore, sample_pdf: Path):
    embedder = FakeEmbedder()
    ingest_file(sample_pdf, store, embedder, Settings(chunk_max_tokens=320))
    result = ingest_file(sample_pdf, store, embedder, Settings(chunk_max_tokens=40))

    assert result.status == "ingested"
    assert store.count("sample.pdf") == result.chunks
