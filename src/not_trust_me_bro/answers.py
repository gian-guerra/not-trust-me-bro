"""Grounded answers: retrieved chunks → an answer that uses only them, with citations
(notes 12, 14, 16, 17).

How grounding is enforced:
- The prompt numbers each chunk [1]..[k] and the model must cite those numbers. Numbers are
  easier for a model to copy exactly than file names, and every one can be checked: a
  citation to a number we never gave is caught (`invalid_citations`).
- If the chunks don't answer the question, the model must reply with one fixed sentence,
  so a refusal can be detected in code instead of guessed.
- Low temperature (note 14): we want the most likely wording of what the sources say,
  not creative variation.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass, field

import litellm

from not_trust_me_bro.retrieval import RetrievedChunk

REFUSAL = "I can't find this in your sources."

SYSTEM_PROMPT = f"""You answer questions for a student using ONLY the numbered sources provided.

Rules:
1. Use only information stated in the sources. Never add facts from your own knowledge, \
even if you are sure they are true.
2. After every sentence that states a fact, cite the source number(s) it comes from, \
like [1] or [2][4]. Cite only numbers that appear in the sources.
3. The sources contain their own academic references, such as (Smith et al., 2020). \
Those point to original studies, not to our sources. You may mention them, but every \
fact still needs our source number in square brackets.
4. If the sources do not contain the answer, reply with exactly this sentence and nothing \
else: {REFUSAL}
5. If the sources answer only part of the question, answer that part and say plainly \
which part they don't cover.
6. Keep the sources' level of certainty. Do not turn "is associated with" into "predicts" \
or "causes", and keep any caveat or conflicting evidence the sources mention. If the \
sources are mixed or tentative, your conclusion must be too.
7. Be clear and concise. Explain in your own words; quote only short phrases.

Example of the citation format:
Leaders receive more attention than other group members [2]. One eye-tracking study \
found observers looked at them three times longer (Foulsham et al., 2010) [2][5]."""


def format_sources(chunks: list[RetrievedChunk]) -> str:
    """Number the chunks and label each with where it comes from."""
    blocks = []
    for i, c in enumerate(chunks, 1):
        header = f"[{i}] {c.citation}" + (f" · § {c.section}" if c.section else "")
        blocks.append(f"{header}\n{c.text}")
    return "\n\n---\n\n".join(blocks)


def build_messages(question: str, chunks: list[RetrievedChunk]) -> list[dict[str, str]]:
    user = f"Sources:\n\n{format_sources(chunks)}\n\n===\n\nQuestion: {question}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


_CITATION = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]")


def cited_numbers(text: str) -> list[int]:
    """[1], [2][3] and [2, 3] → [1, 2, 3], in order of first appearance."""
    numbers: list[int] = []
    for group in _CITATION.findall(text):
        for n in re.split(r"[,;]", group):
            if int(n) not in numbers:
                numbers.append(int(n))
    return numbers


def is_refusal(text: str) -> bool:
    normalize = lambda s: re.sub(r"[^a-z ]", "", s.lower().replace("’", "'"))  # noqa: E731
    return normalize(REFUSAL) in normalize(text)


@dataclass(frozen=True)
class Answer:
    question: str
    text: str
    chunks: list[RetrievedChunk]  # everything the model was shown, in prompt order
    cited: list[RetrievedChunk] = field(default_factory=list)  # the ones it cited
    invalid_citations: list[int] = field(default_factory=list)  # numbers it made up
    refused: bool = False

    @property
    def uncited(self) -> bool:
        """An answer that claims something but points to none of our sources: not grounded."""
        return not self.refused and not self.cited


# (model, messages, **kwargs) → text. Swappable so tests don't need an LLM.
Complete = Callable[..., str]


def litellm_complete(model: str, messages: list[dict[str, str]], **kwargs) -> str:
    response = litellm.completion(model=model, messages=messages, **kwargs)
    return response.choices[0].message.content or ""


def answer(
    question: str,
    chunks: list[RetrievedChunk],
    model: str,
    temperature: float = 0.1,
    think: bool = False,
    api_key: str | None = None,
    complete: Complete = litellm_complete,
) -> Answer:
    if not chunks:  # nothing retrieved: no LLM call, nothing to ground an answer in
        return Answer(question, REFUSAL, [], refused=True)

    kwargs: dict = {"temperature": temperature}
    if model.startswith(("ollama", "ollama_chat")):
        kwargs["think"] = think  # local thinking models (qwen3, gemma4) only
    if api_key:
        kwargs["api_key"] = api_key

    text = complete(model, build_messages(question, chunks), **kwargs).strip()
    numbers = cited_numbers(text)
    return Answer(
        question=question,
        text=text,
        chunks=chunks,
        cited=[chunks[n - 1] for n in numbers if 1 <= n <= len(chunks)],
        invalid_citations=[n for n in numbers if not 1 <= n <= len(chunks)],
        refused=is_refusal(text),
    )
