"""Chunking: split a document's pages into the units we embed and retrieve (notes 08, 09).

Strategy: structure-aware packing, bounded by size.
1. Pages → blocks: paragraphs, headings and tables. A sentence cut by a page break is
   rejoined (skipping any table or caption the layout put at the top of the next page),
   so the paragraph becomes one block spanning both pages.
2. Blocks → chunks: pack whole blocks up to `max_tokens`. A heading starts a new chunk
   (sections are natural topic boundaries) unless the current chunk is still tiny.
   A block too big on its own is split: tables by rows (repeating the header row),
   text by sentences.
3. Overlap: a chunk starts with the last sentences (up to `overlap_tokens`) of the previous
   chunk in the same section, so an idea cut at a boundary is still whole in one of them.

Sizes are approximate tokens, because embedding models cap their input in tokens and
silently truncate the rest. Words are a bad proxy: prose is ~1.3 tokens per word, but a
Markdown table row is 4-10 (`|0.20|−0.07|` has almost no spaces). PER-9 checks the estimate
against the real tokenizer.
"""

import re
from dataclasses import dataclass
from typing import Literal

from not_trust_me_bro.ingestion import Page


@dataclass(frozen=True)
class Chunk:
    source: str
    chunk_index: int  # position within the document
    text: str
    page_start: int
    page_end: int
    section: str | None  # the section holding most of the chunk's content


@dataclass(frozen=True)
class _Block:
    kind: Literal["text", "heading", "table"]
    text: str
    page_start: int
    page_end: int


_TOKEN = re.compile(r"\w+|[^\w\s]")


def count_tokens(text: str) -> int:
    """Approximate token count: words and punctuation marks counted separately,
    close to how BERT-style tokenizers split English."""
    return len(_TOKEN.findall(text))


# --- 1. pages → blocks ----------------------------------------------------------------------

_ENDS_SENTENCE = re.compile(r'[.!?:"”’)\]]\s*$')
# "TABLE 1 Included studies.", "**Figure 2.**": floats that a layout engine puts at a page top
_CAPTION = re.compile(r"^[#*_\s]*(table|fig\.?|figure)\s*\d", re.IGNORECASE)


def _kind(paragraph: str) -> Literal["text", "heading", "table"]:
    if paragraph.startswith("#"):
        return "heading"
    if paragraph.startswith("|"):
        return "table"
    return "text"


def to_blocks(pages: list[Page]) -> list[_Block]:
    blocks: list[_Block] = []
    for page in pages:
        # Index of the previous page's last paragraph, if it stopped mid-sentence.
        last = len(blocks) - 1
        open_idx = (
            last
            if blocks
            and blocks[last].kind == "text"
            and blocks[last].page_end == page.page_number - 1
            and not _ENDS_SENTENCE.search(blocks[last].text)
            else None
        )
        for para in (p.strip() for p in page.text.split("\n\n")):
            if not para:
                continue
            kind = _kind(para)
            if open_idx is not None and kind == "text" and not _CAPTION.match(para):
                # The sentence continues here: rejoin it, even if a table or caption sat between.
                prev = blocks[open_idx]
                text = f"{prev.text} {para}"
                blocks[open_idx] = _Block("text", text, prev.page_start, page.page_number)
                open_idx = None
                continue
            if kind == "heading":
                open_idx = None  # a new section: whatever was open is not continued
            blocks.append(_Block(kind, para, page.page_number, page.page_number))
    return blocks


# --- splitting oversized blocks ---------------------------------------------------------------

# Split after . ! ? when the next sentence starts with a capital or a quote.
# Abbreviations common in academic text are protected first.
_ABBREVIATIONS = re.compile(r"\b(et al|e\.g|i\.e|vs|cf|Fig|Figs|No|Vol|pp|Dr|Mr|Ms)\.")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z“\"‘'])")
_PLACEHOLDER = "\x00"


def split_sentences(text: str) -> list[str]:
    protected = _ABBREVIATIONS.sub(lambda m: m.group(0).replace(".", _PLACEHOLDER), text)
    return [s.replace(_PLACEHOLDER, ".") for s in _SENTENCE_END.split(protected) if s.strip()]


def _pack(units: list[str], max_tokens: int, joiner: str) -> list[str]:
    """Greedily pack `units` into pieces of at most `max_tokens`.
    A unit too big on its own (e.g. a 400-token sentence) is cut at word boundaries."""
    pieces: list[str] = []
    current: list[str] = []
    for unit in units:
        parts = [unit] if count_tokens(unit) <= max_tokens else _pack(unit.split(), max_tokens, " ")
        for part in parts:
            if current and count_tokens(joiner.join([*current, part])) > max_tokens:
                pieces.append(joiner.join(current))
                current = []
            current.append(part)
    if current:
        pieces.append(joiner.join(current))
    return pieces


def split_block(block: _Block, max_tokens: int) -> list[_Block]:
    if count_tokens(block.text) <= max_tokens:
        return [block]
    if block.kind == "table":
        # Keep the header row and the |---| separator on every piece, so each stays readable.
        lines = block.text.splitlines()
        header, rows = lines[:2], lines[2:]
        budget = max_tokens - count_tokens("\n".join(header))
        texts = ["\n".join([*header, rows_]) for rows_ in _pack(rows, budget, "\n")]
    else:
        texts = _pack(split_sentences(block.text), max_tokens, " ")
    return [_Block(block.kind, t, block.page_start, block.page_end) for t in texts]


# --- 2 & 3. blocks → chunks -------------------------------------------------------------------


def _heading_text(heading: str) -> str:
    return re.sub(r"[#*_]", "", heading).strip()


def _tail_sentences(block: _Block, budget: int) -> list[_Block]:
    """The last sentences of a text block, up to `budget` tokens, used as overlap."""
    if block.kind != "text" or budget <= 0:
        return []
    tail: list[str] = []
    for sentence in reversed(split_sentences(block.text)):
        if count_tokens(" ".join([sentence, *tail])) > budget:
            break
        tail.insert(0, sentence)
    return [_Block("text", " ".join(tail), block.page_start, block.page_end)] if tail else []


def chunk_pages(
    pages: list[Page], max_tokens: int = 320, overlap_tokens: int = 75, min_tokens: int = 65
) -> list[Chunk]:
    """Defaults come from the starter corpus (prose ≈ 1.28 tokens per word):
    - max 320 ≈ 250 words: paragraphs have a median of ~110 words, so a chunk holds 1-3.
    - overlap 75 ≈ 60 words: 90% of sentences are ≤ 50 words, so overlap usually carries
      at least one whole sentence.
    - min 65 ≈ 50 words: a smaller section is merged into the next one instead of
      becoming its own chunk.
    Headings are added on top of a full piece, so a chunk can exceed `max_tokens` by a
    heading's length (a few tokens).
    """
    if not pages:
        return []
    source = pages[0].source
    chunks: list[Chunk] = []
    current: list[_Block] = []  # the chunk being built: overlap, headings, content
    content_tokens = 0  # tokens of real content in `current`: not overlap, not headings
    section: str | None = None
    tokens_per_section: dict[str | None, int] = {}

    def size_with(piece: _Block) -> int:
        return count_tokens("\n\n".join(b.text for b in [*current, piece]))

    def drop_overlap() -> None:
        nonlocal current
        current = [b for b in current if b.kind == "heading"]

    def emit(keep_overlap: bool) -> None:
        nonlocal current, content_tokens
        # Headings at the end belong to the content that follows them: move them forward.
        trailing: list[_Block] = []
        while current and current[-1].kind == "heading":
            trailing.insert(0, current.pop())
        if content_tokens > 0:  # never emit a chunk that is only overlap and/or headings
            chunks.append(
                Chunk(
                    source=source,
                    chunk_index=len(chunks),
                    text="\n\n".join(b.text for b in current),
                    page_start=min(b.page_start for b in current),
                    page_end=max(b.page_end for b in current),
                    section=max(tokens_per_section, key=tokens_per_section.__getitem__),
                )
            )
        if trailing:  # a new section starts: no overlap from the old one
            current = trailing
        elif keep_overlap and current:
            current = _tail_sentences(current[-1], overlap_tokens)
        else:
            current = []
        content_tokens = 0
        tokens_per_section.clear()

    for block in to_blocks(pages):
        if block.kind == "heading":
            if content_tokens >= min_tokens:  # close the previous section's chunk
                emit(keep_overlap=False)
            elif content_tokens == 0:  # overlap from the previous section doesn't belong here
                drop_overlap()
            # A tiny section (< min_tokens) stays in the current chunk; its heading goes inline.
            section = _heading_text(block.text)
            current.append(block)
            continue

        for piece in split_block(block, max_tokens):
            if content_tokens > 0 and size_with(piece) > max_tokens:
                emit(keep_overlap=True)
            if content_tokens == 0 and size_with(piece) > max_tokens:
                drop_overlap()  # the piece fills the chunk by itself
            n = count_tokens(piece.text)
            current.append(piece)
            content_tokens += n
            tokens_per_section[section] = tokens_per_section.get(section, 0) + n

    emit(keep_overlap=False)
    return chunks
