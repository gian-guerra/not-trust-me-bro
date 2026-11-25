"""The whole system behind one object: ingest documents, ask questions.

This is the layer the CLI (and later the API) talks to. It owns the expensive resources —
the Weaviate connection and the embedding model — so they are created once per session,
not once per file or per question:

    with open_rag(get_settings()) as rag:
        rag.ingest(Path("data/paper.pdf"))
        print(rag.ask("Why do people defer to high-status people?").text)
"""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Protocol

from not_trust_me_bro.answers import Answer, answer
from not_trust_me_bro.config import Settings
from not_trust_me_bro.embeddings import Embedder
from not_trust_me_bro.pipeline import DocumentEmbedder, IngestResult, ingest_file
from not_trust_me_bro.retrieval import Mode, QueryEmbedder, Retriever
from not_trust_me_bro.store import ChunkStore, collection_name, connect


def find_pdfs(path: Path) -> list[Path]:
    """One PDF, or every PDF under a folder (recursively), in a stable order."""
    if path.is_file():
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"not a PDF: {path}")
        return [path]
    if path.is_dir():
        return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() == ".pdf")
    raise FileNotFoundError(path)


class Embeds(DocumentEmbedder, QueryEmbedder, Protocol): ...


class Rag:
    def __init__(self, store: ChunkStore, settings: Settings, embedder: Embeds | None = None):
        self.store = store
        self.settings = settings
        # The real embedder loads its model on first use, so `ntmb sources` never pays for it.
        self.embedder = embedder or Embedder(
            settings.embedding_model, settings.embedding_query_prefix
        )

    def ingest(self, path: Path, force: bool = False) -> IngestResult:
        return ingest_file(path, self.store, self.embedder, self.settings, force=force)

    def sources(self) -> dict[str, int]:
        return self.store.sources()

    def ask(
        self,
        question: str,
        sources: list[str] | None = None,
        k: int | None = None,
        mode: Mode | None = None,
        model: str | None = None,
        think: bool | None = None,
    ) -> Answer:
        """Retrieve the top-k chunks, then answer from them. Unset options use the settings."""
        s = self.settings
        overrides = {
            "retrieval_k": k,
            "retrieval_mode": mode,
            "llm_model": model,
            "llm_think": think,
        }
        s = s.model_copy(update={key: v for key, v in overrides.items() if v is not None})

        chunks = Retriever(self.store, self.embedder).retrieve(
            question,
            k=s.retrieval_k,
            mode=s.retrieval_mode,
            alpha=s.retrieval_alpha,
            fusion=s.retrieval_fusion,
            sources=sources,
        )
        return answer(
            question, chunks, s.llm_model, s.llm_temperature, s.llm_think, s.llm_api_key()
        )


@contextmanager
def open_rag(settings: Settings) -> Iterator[Rag]:
    with connect(settings) as client:
        yield Rag(ChunkStore(client, collection_name(settings.embedding_model)), settings)
