import os

import pytest

from not_trust_me_bro.answers import (
    REFUSAL,
    SYSTEM_PROMPT,
    answer,
    build_messages,
    cited_numbers,
    format_sources,
    is_refusal,
)
from not_trust_me_bro.retrieval import RetrievedChunk


def chunk(source: str, page: int, text: str, section: str = "Intro") -> RetrievedChunk:
    return RetrievedChunk(text, source, page, page, section, chunk_index=0, score=1.0)


CHUNKS = [
    chunk("a.pdf", 3, "Leaders receive more attention."),
    chunk("b.pdf", 7, "Testosterone is associated with status, not necessarily causally."),
]


class FakeLLM:
    """Stands in for litellm: returns a fixed reply and records what it was sent."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[dict] = []

    def __call__(self, model, messages, **kwargs) -> str:
        self.calls.append({"model": model, "messages": messages, **kwargs})
        return self.reply


# --- prompt ---------------------------------------------------------------------------------


def test_sources_are_numbered_and_labelled_with_their_citation():
    text = format_sources(CHUNKS)
    assert text.startswith("[1] a.pdf, p. 3 · § Intro\nLeaders receive more attention.")
    assert "[2] b.pdf, p. 7 · § Intro" in text


def test_messages_hold_the_rules_the_sources_and_the_question():
    system, user = build_messages("Why?", CHUNKS)
    assert system == {"role": "system", "content": SYSTEM_PROMPT}
    assert REFUSAL in SYSTEM_PROMPT
    assert "[2] b.pdf, p. 7" in user["content"]
    assert user["content"].endswith("Question: Why?")


# --- parsing --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("A [1]. B [2][3].", [1, 2, 3]),
        ("A [2, 1]. B [1].", [2, 1]),
        ("A (Smith et al., 2020) [1].", [1]),  # academic references are not our citations
        ("No citations here.", []),
    ],
)
def test_cited_numbers(text, expected):
    assert cited_numbers(text) == expected


def test_refusal_is_detected_despite_small_variations():
    assert is_refusal("I can’t find this in your sources.")  # curly apostrophe
    assert is_refusal("i can't find this in your sources")
    assert not is_refusal("I can find this in your sources [1].")


# --- answer ---------------------------------------------------------------------------------


def test_citations_are_mapped_back_to_the_chunks():
    a = answer("q", CHUNKS, "ollama_chat/m", complete=FakeLLM("Status matters [2]. Also [1]."))
    assert [c.source for c in a.cited] == ["b.pdf", "a.pdf"]
    assert not a.refused and not a.uncited


def test_citation_numbers_that_were_never_given_are_caught():
    a = answer("q", CHUNKS, "ollama_chat/m", complete=FakeLLM("Claim [1]. Made up [7]."))
    assert a.invalid_citations == [7]
    assert [c.source for c in a.cited] == ["a.pdf"]


def test_refusal_is_flagged():
    a = answer("q", CHUNKS, "ollama_chat/m", complete=FakeLLM(REFUSAL))
    assert a.refused and not a.uncited


def test_an_answer_without_our_citations_is_flagged_uncited():
    reply = "Leaders get attention (Foulsham et al., 2010)."  # its own style, not [n]
    a = answer("q", CHUNKS, "ollama_chat/m", complete=FakeLLM(reply))
    assert a.uncited


def test_no_chunks_means_no_llm_call_and_a_refusal():
    llm = FakeLLM("should not be called")
    a = answer("q", [], "ollama_chat/m", complete=llm)
    assert a.refused and llm.calls == []


def test_think_is_sent_to_local_models_only():
    local, remote = FakeLLM(REFUSAL), FakeLLM(REFUSAL)
    answer("q", CHUNKS, "ollama_chat/qwen3:14b", think=True, complete=local)
    answer("q", CHUNKS, "openai/gpt-4o-mini", think=True, api_key="sk-x", complete=remote)
    assert local.calls[0]["think"] is True
    assert "think" not in remote.calls[0] and remote.calls[0]["api_key"] == "sk-x"


# --- real local model (opt-in: NTMB_TEST_LLM=1, needs Ollama) --------------------------------


@pytest.mark.skipif(not os.getenv("NTMB_TEST_LLM"), reason="set NTMB_TEST_LLM=1 to run")
def test_real_model_answers_with_citations_and_refuses_outside_knowledge():
    from not_trust_me_bro.config import Settings

    model = Settings().llm_model
    grounded = answer("Why do leaders get attention?", CHUNKS, model)
    tempted = answer("What did Freud say about the Oedipus complex?", CHUNKS, model)

    assert grounded.cited and not grounded.invalid_citations
    assert tempted.refused
