"""Uses the real default model (downloaded once to the Hugging Face cache, ~130 MB)."""

import math

import pytest

from not_trust_me_bro.config import Settings
from not_trust_me_bro.embeddings import Embedder, TruncationError


@pytest.fixture(scope="module")
def embedder() -> Embedder:
    s = Settings()
    return Embedder(s.embedding_model, s.embedding_query_prefix)


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def test_vectors_are_normalized_and_have_the_model_dimensions(embedder: Embedder):
    [vector] = embedder.embed_documents(["Power changes how people attend to others."])
    assert len(vector) == embedder.dimensions == 384
    assert math.isclose(dot(vector, vector), 1.0, abs_tol=1e-4)


def test_similar_meaning_scores_higher_than_unrelated_text(embedder: Embedder):
    status, refund, power = embedder.embed_documents(
        [
            "High-status people receive more attention from others.",
            "Our store refunds unused items within 30 days.",
            "People watch those at the top of the hierarchy closely.",
        ]
    )
    assert dot(status, power) > dot(status, refund)


def test_text_over_the_limit_is_refused_instead_of_silently_truncated(embedder: Embedder):
    too_long = "status " * (embedder.max_tokens + 10)
    with pytest.raises(TruncationError):
        embedder.embed_documents(["short is fine", too_long])


def test_query_gets_the_instruction_prefix_and_documents_do_not(embedder: Embedder):
    text = "Why do people defer to leaders?"
    [as_document] = embedder.embed_documents([text])
    as_query = embedder.embed_query(text)
    assert as_document != as_query
