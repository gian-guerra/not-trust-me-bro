"""The library layer end to end: a real PDF, real Weaviate, a fake embedder and a fake LLM."""

import uuid
from pathlib import Path
from types import SimpleNamespace

import litellm
import pytest

from not_trust_me_bro.config import Settings
from not_trust_me_bro.rag import Rag, find_pdfs
from not_trust_me_bro.store import ChunkStore, connect


class FakeEmbedder:
    """Every text gets the same vector: enough to test the plumbing, not the ranking."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0] for _ in texts]

    def embed_query(self, query: str) -> list[float]:
        return [1.0, 0.0, 0.0]


# --- find_pdfs ------------------------------------------------------------------------------


def test_find_pdfs_searches_folders_recursively_and_ignores_other_files(tmp_path: Path):
    for name in ["b.pdf", "notes.txt", "sub/a.PDF"]:
        (tmp_path / name).parent.mkdir(exist_ok=True)
        (tmp_path / name).touch()
    assert [p.name for p in find_pdfs(tmp_path)] == ["b.pdf", "a.PDF"]


def test_find_pdfs_rejects_missing_paths_and_non_pdfs(tmp_path: Path):
    (tmp_path / "notes.txt").touch()
    with pytest.raises(FileNotFoundError):
        find_pdfs(tmp_path / "missing.pdf")
    with pytest.raises(ValueError):
        find_pdfs(tmp_path / "notes.txt")


# --- Rag (needs Weaviate) -------------------------------------------------------------------


@pytest.fixture
def rag():
    try:
        client = connect(Settings())
        client.is_ready()
    except Exception:
        pytest.skip("Weaviate is not running (docker compose up -d)")
    store = ChunkStore(client, f"Test_{uuid.uuid4().hex[:8]}")
    yield Rag(store, Settings(), embedder=FakeEmbedder())
    client.collections.delete(store.name)
    client.close()


def test_ingest_then_ask_answers_from_the_document(rag: Rag, sample_pdf: Path, monkeypatch):
    sent = {}

    def fake_completion(model, messages, **kwargs):
        sent.update(model=model, prompt=messages[1]["content"])
        reply = SimpleNamespace(content="Hierarchies form quickly [1].")
        return SimpleNamespace(choices=[SimpleNamespace(message=reply)])

    monkeypatch.setattr(litellm, "completion", fake_completion)

    assert rag.ingest(sample_pdf).status == "ingested"
    assert rag.ingest(sample_pdf).status == "unchanged"
    assert rag.sources() == {"sample.pdf": rag.store.count("sample.pdf")}

    a = rag.ask("How fast do hierarchies form?", k=2, mode="semantic", model="openai/x")
    assert sent["model"] == "openai/x"  # per-question override beats the settings
    assert "sample.pdf, p. 1" in sent["prompt"]
    assert a.cited[0].source == "sample.pdf"
