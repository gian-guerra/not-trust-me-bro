"""Retrieval against a real Weaviate (docker compose up -d) in a throwaway collection.

Vectors are hand-made so each mode's behaviour is predictable: the fake query embedder always
points at the "leaders" chunk, while the words of the question may point elsewhere.
"""

import uuid

import pytest

from not_trust_me_bro.chunking import Chunk
from not_trust_me_bro.config import Settings
from not_trust_me_bro.retrieval import Retriever
from not_trust_me_bro.store import ChunkStore, connect

LEADERS = "Group members pay close attention to their leaders."
MTQ = "The MTQ-48 questionnaire measures mental toughness in athletes."
REFUND = "A battery cannot be refunded once it has been used."


class FakeQueryEmbedder:
    def embed_query(self, query: str) -> list[float]:
        return [1.0, 0.0, 0.0]  # always closest to the LEADERS chunk


@pytest.fixture(scope="module")
def retriever():
    try:
        client = connect(Settings())
        client.is_ready()
    except Exception:
        pytest.skip("Weaviate is not running (docker compose up -d)")
    store = ChunkStore(client, f"Test_{uuid.uuid4().hex[:8]}")
    store.replace_document(
        [
            Chunk("psych.pdf", 0, LEADERS, 3, 3, "Attention"),
            Chunk("psych.pdf", 1, MTQ, 7, 8, "Methods"),
        ],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        "fp",
    )
    store.replace_document([Chunk("shop.pdf", 0, REFUND, 1, 1, "Refunds")], [[0.0, 0.0, 1.0]], "fp")
    yield Retriever(store, FakeQueryEmbedder())
    client.collections.delete(store.name)
    client.close()


def texts(results) -> list[str]:
    return [r.text for r in results]


def test_semantic_follows_the_vector_even_when_words_disagree(retriever: Retriever):
    assert texts(retriever.retrieve("MTQ-48", k=1, mode="semantic")) == [LEADERS]


def test_bm25_follows_the_exact_words(retriever: Retriever):
    assert texts(retriever.retrieve("MTQ-48", k=1, mode="bm25")) == [MTQ]


@pytest.mark.parametrize(("alpha", "expected"), [(1.0, LEADERS), (0.0, MTQ)])
def test_alpha_moves_hybrid_between_semantic_and_bm25(retriever: Retriever, alpha, expected):
    assert texts(retriever.retrieve("MTQ-48", k=1, mode="hybrid", alpha=alpha))[0] == expected


def test_rrf_fusion_is_available(retriever: Retriever):
    results = retriever.retrieve("MTQ-48", k=2, mode="hybrid", fusion="ranked")
    assert set(texts(results)) == {LEADERS, MTQ}


def test_results_are_best_first_with_k_respected(retriever: Retriever):
    results = retriever.retrieve("leaders attention", k=2, mode="semantic")
    assert len(results) == 2
    assert results[0].score >= results[1].score


def test_source_filter_restricts_the_search(retriever: Retriever):
    results = retriever.retrieve("battery refund", k=3, mode="bm25", sources=["psych.pdf"])
    assert all(r.source == "psych.pdf" for r in results)
    assert REFUND not in texts(results)


def test_results_carry_what_a_citation_needs(retriever: Retriever):
    [hit] = retriever.retrieve("MTQ-48", k=1, mode="bm25")
    assert (hit.source, hit.page_start, hit.page_end, hit.section) == ("psych.pdf", 7, 8, "Methods")
    assert hit.citation == "psych.pdf, p. 7–8"
