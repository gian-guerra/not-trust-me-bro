"""Retrieval: question → the top-k most relevant chunks (notes 02, 03, 04).

Three modes over the same stored objects:
- "semantic": embed the question, nearest vectors in the HNSW index. Matches meaning.
- "bm25":     keyword scoring over the text index. Matches exact words.
- "hybrid":   both, fused into one ranking. `alpha` weights them (note 04's β):
              1.0 = pure semantic, 0.0 = pure BM25.

Scores are only comparable within one mode: semantic → cosine similarity (−1..1),
bm25 → unbounded BM25 score, hybrid → fused score (0..1).

Note: the embedder adds BGE's query instruction to the vector side only; BM25 always gets
the plain question, because the instruction's words would pollute keyword matching.
"""

from dataclasses import dataclass
from typing import Literal, Protocol

from weaviate.classes.query import Filter, HybridFusion, MetadataQuery

from not_trust_me_bro.store import ChunkStore

Mode = Literal["semantic", "bm25", "hybrid"]
Fusion = Literal["relative_score", "ranked"]

_FUSION = {
    # Normalizes each list's scores to 0..1, then adds them weighted by alpha (Weaviate default).
    "relative_score": HybridFusion.RELATIVE_SCORE,
    # Reciprocal Rank Fusion: uses ranks, not scores (note 04).
    "ranked": HybridFusion.RANKED,
}


class QueryEmbedder(Protocol):
    def embed_query(self, query: str) -> list[float]: ...


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    source: str
    page_start: int
    page_end: int
    section: str
    chunk_index: int
    score: float  # only comparable within one mode, see module docstring

    @property
    def citation(self) -> str:
        pages = f"p. {self.page_start}" + (
            f"–{self.page_end}" if self.page_end != self.page_start else ""
        )
        return f"{self.source}, {pages}"


class Retriever:
    def __init__(self, store: ChunkStore, embedder: QueryEmbedder) -> None:
        self.collection = store.collection
        self.embedder = embedder

    def retrieve(
        self,
        question: str,
        k: int = 5,
        mode: Mode = "hybrid",
        alpha: float = 0.5,
        fusion: Fusion = "relative_score",
        sources: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        """Return the k best chunks for `question`, best first.
        `sources` restricts the search to those documents (a metadata filter, note 04)."""
        filters = Filter.by_property("source").contains_any(sources) if sources else None
        query = self.collection.query

        if mode == "semantic":
            result = query.near_vector(
                self.embedder.embed_query(question),
                limit=k,
                filters=filters,
                return_metadata=MetadataQuery(distance=True),
            )
            # Weaviate returns cosine *distance* (1 − similarity); flip it so higher = better.
            scores = [1 - o.metadata.distance for o in result.objects]
        elif mode == "bm25":
            result = query.bm25(
                question, limit=k, filters=filters, return_metadata=MetadataQuery(score=True)
            )
            scores = [o.metadata.score for o in result.objects]
        elif mode == "hybrid":
            result = query.hybrid(
                question,
                vector=self.embedder.embed_query(question),
                alpha=alpha,
                fusion_type=_FUSION[fusion],
                limit=k,
                filters=filters,
                return_metadata=MetadataQuery(score=True),
            )
            scores = [o.metadata.score for o in result.objects]
        else:
            raise ValueError(f"unknown retrieval mode: {mode!r}")

        return [
            RetrievedChunk(
                text=o.properties["text"],
                source=o.properties["source"],
                page_start=o.properties["page_start"],
                page_end=o.properties["page_end"],
                section=o.properties["section"],
                chunk_index=o.properties["chunk_index"],
                score=score,
            )
            for o, score in zip(result.objects, scores, strict=True)
        ]
