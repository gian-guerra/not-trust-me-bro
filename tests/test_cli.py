"""The CLI is tested against a fake library: these tests check parsing and printing only."""

from contextlib import contextmanager
from pathlib import Path

import pytest
from typer.testing import CliRunner
from weaviate.exceptions import WeaviateStartUpError

from not_trust_me_bro import cli
from not_trust_me_bro.answers import Answer
from not_trust_me_bro.pipeline import IngestResult
from not_trust_me_bro.retrieval import RetrievedChunk

runner = CliRunner()

CHUNKS = [
    RetrievedChunk("Leaders get attention.", "a.pdf", 3, 3, "Intro", 0, 0.9),
    RetrievedChunk("Status is sticky.", "b.pdf", 7, 8, "Methods", 0, 0.5),
]


class FakeRag:
    def __init__(self, answer_text: str = "Leaders get attention [1].") -> None:
        self.answer_text = answer_text
        self.asked: dict = {}

    def ingest(self, path: Path, force: bool = False) -> IngestResult:
        if path.name == "broken.pdf":
            raise RuntimeError("cannot open broken document")
        status = "unchanged" if path.name == "old.pdf" else "ingested"
        return IngestResult(path.name, status, 12)

    def sources(self) -> dict[str, int]:
        return {"a.pdf": 10, "b.pdf": 20}

    def ask(self, question: str, **kwargs) -> Answer:
        self.asked = {"question": question, **kwargs}
        cited = [CHUNKS[0]] if "[1]" in self.answer_text else []
        return Answer(question, self.answer_text, CHUNKS, cited=cited)


@pytest.fixture
def fake(monkeypatch) -> FakeRag:
    rag = FakeRag()

    @contextmanager
    def open_fake(settings):
        yield rag

    monkeypatch.setattr(cli, "open_rag", open_fake)
    return rag


def test_config_masks_secrets(monkeypatch):
    monkeypatch.setenv("NTMB_OPENAI_API_KEY", "sk-test-not-real")
    cli.get_settings.cache_clear()
    result = runner.invoke(cli.app, ["config"])

    assert result.exit_code == 0
    assert "llm_model" in result.output
    assert "sk-test-not-real" not in result.output


# --- ingest ---------------------------------------------------------------------------------


def test_ingest_reports_every_file_and_keeps_going_after_a_failure(fake, tmp_path: Path):
    for name in ["new.pdf", "old.pdf", "broken.pdf"]:
        (tmp_path / name).touch()
    result = runner.invoke(cli.app, ["ingest", str(tmp_path)])

    out = result.output
    assert "[1/3] broken.pdf\n      ✗ failed: cannot open broken document" in out
    assert "[2/3] new.pdf\n      ✓ ingested: 12 chunks" in out
    assert "[3/3] old.pdf\n      = unchanged since last ingest (12 chunks)" in out
    assert "2/3 documents ready" in result.output
    assert result.exit_code == 1


def test_ingest_a_missing_path_fails_clearly(fake, tmp_path: Path):
    result = runner.invoke(cli.app, ["ingest", str(tmp_path / "nope.pdf")])
    assert result.exit_code == 1
    assert "Can't ingest" in result.output


# --- ask ------------------------------------------------------------------------------------


def test_ask_prints_the_answer_and_marks_the_cited_sources(fake):
    result = runner.invoke(cli.app, ["ask", "Why?", "-s", "a.pdf", "--k", "3", "--think"])

    assert result.exit_code == 0
    assert "Leaders get attention [1]." in result.output
    assert "★ [1] a.pdf, p. 3  § Intro" in result.output
    assert "  [2] b.pdf, p. 7–8" in result.output
    assert fake.asked == {
        "question": "Why?",
        "sources": ["a.pdf"],
        "k": 3,
        "mode": None,
        "model": None,
        "think": True,
    }


def test_ask_warns_when_the_answer_cites_nothing(fake):
    fake.answer_text = "Leaders get attention (Smith, 2020)."
    result = runner.invoke(cli.app, ["ask", "Why?"])
    assert "No citations to your sources" in result.output


def test_ask_rejects_a_source_that_is_not_in_the_library(fake):
    result = runner.invoke(cli.app, ["ask", "Why?", "-s", "missing.pdf"])
    assert result.exit_code == 1
    assert "Not in your library: missing.pdf" in result.output


# --- sources / errors -----------------------------------------------------------------------


def test_sources_lists_documents(fake):
    result = runner.invoke(cli.app, ["sources"])
    assert "a.pdf  (10 chunks)" in result.output


def test_weaviate_down_gives_a_readable_message(monkeypatch):
    def down(settings):
        raise WeaviateStartUpError("connection refused")

    monkeypatch.setattr(cli, "open_rag", down)
    result = runner.invoke(cli.app, ["sources"])
    assert result.exit_code == 1
    assert "docker compose up -d" in result.output
