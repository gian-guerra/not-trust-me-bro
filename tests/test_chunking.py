from not_trust_me_bro.chunking import (
    _Block,
    chunk_pages,
    count_tokens,
    split_block,
    split_sentences,
    to_blocks,
)
from not_trust_me_bro.ingestion import Page


def pages(*texts: str) -> list[Page]:
    return [Page(source="doc.pdf", page_number=i, text=t) for i, t in enumerate(texts, 1)]


def sentence(n: int) -> str:
    """A distinct sentence of 11 tokens (10 words + the full stop)."""
    return f"Sentence {n} talks about status and power in small groups."


def paragraph(start: int, count: int) -> str:
    return " ".join(sentence(i) for i in range(start, start + count))


# --- count_tokens ---------------------------------------------------------------------------


def test_count_tokens_sees_through_tables_that_have_no_spaces():
    row = "|1|Papageorgiou et al. (2017)|608|MTQ-48|SD3|0.20|"
    assert len(row.split()) == 4
    assert count_tokens(row) > 20


# --- split_sentences ------------------------------------------------------------------------


def test_split_sentences_keeps_citations_and_abbreviations_intact():
    text = "Leaders get deference (Smith et al., 2020). Others, e.g. peers, follow. See Fig. 2 now."
    assert split_sentences(text) == [
        "Leaders get deference (Smith et al., 2020).",
        "Others, e.g. peers, follow.",
        "See Fig. 2 now.",
    ]


# --- to_blocks: page breaks -----------------------------------------------------------------


def test_sentence_cut_by_a_page_break_is_rejoined_across_both_pages():
    blocks = to_blocks(pages("Status matters because people", "notice it quickly. Next idea."))
    assert len(blocks) == 1
    assert blocks[0].text == "Status matters because people notice it quickly. Next idea."
    assert (blocks[0].page_start, blocks[0].page_end) == (1, 2)


def test_rejoin_skips_a_table_and_caption_at_the_top_of_the_next_page():
    blocks = to_blocks(
        pages(
            "We used a random-effects",
            "TABLE 1 Included studies.\n\n|K|n|\n|---|---|\n|1|608|\n\nmodel for all analyses.",
        )
    )
    assert [b.text for b in blocks] == [
        "We used a random-effects model for all analyses.",
        "TABLE 1 Included studies.",
        "|K|n|\n|---|---|\n|1|608|",
    ]


def test_finished_sentence_or_new_heading_is_not_rejoined():
    assert len(to_blocks(pages("A full sentence.", "Another one."))) == 2
    assert len(to_blocks(pages("Cut mid", "## New section\n\nText."))) == 3


# --- split_block ----------------------------------------------------------------------------


def test_big_table_is_split_by_rows_and_every_piece_repeats_the_header():
    rows = [f"|{i}|Author {i} et al. (2020)|0.{i:02d}|" for i in range(40)]
    table = _Block("table", "\n".join(["|K|Study|r|", "|---|---|---|", *rows]), 6, 6)
    pieces = split_block(table, max_tokens=120)

    assert len(pieces) > 1
    assert all(p.text.startswith("|K|Study|r|\n|---|---|---|") for p in pieces)
    assert all(count_tokens(p.text) <= 120 for p in pieces)
    assert sum(p.text.count("et al.") for p in pieces) == 40  # no row lost or duplicated


def test_long_paragraph_is_split_at_sentence_boundaries():
    block = _Block("text", paragraph(0, 10), 1, 1)  # 100 tokens
    pieces = split_block(block, max_tokens=35)
    assert all(p.text.endswith(".") for p in pieces)
    assert all(count_tokens(p.text) <= 35 for p in pieces)


# --- chunk_pages ----------------------------------------------------------------------------


def test_chunks_respect_the_size_limit_and_keep_page_numbers():
    two_pages = pages(paragraph(0, 20), paragraph(20, 20))
    chunks = chunk_pages(two_pages, max_tokens=60, overlap_tokens=0)
    assert all(count_tokens(c.text) <= 60 for c in chunks)
    assert chunks[0].page_start == 1
    assert chunks[-1].page_end == 2
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_overlap_repeats_the_last_sentence_within_a_section():
    chunks = chunk_pages(pages(paragraph(0, 3), paragraph(3, 3)), max_tokens=50, overlap_tokens=12)
    assert chunks[1].text.startswith(sentence(2))  # last sentence of chunk 0
    assert sentence(2) in chunks[0].text


def test_overlap_is_dropped_when_the_next_piece_fills_the_chunk_alone():
    chunks = chunk_pages(pages(paragraph(0, 3), paragraph(3, 3)), max_tokens=40, overlap_tokens=12)
    assert chunks[1].text.startswith(sentence(3))
    assert all(count_tokens(c.text) <= 40 for c in chunks)


def test_a_heading_starts_a_new_chunk_without_overlap_from_the_old_section():
    text = f"## Power\n\n{paragraph(0, 8)}\n\n## Status\n\n{paragraph(8, 8)}"
    chunks = chunk_pages(pages(text), max_tokens=200, overlap_tokens=30, min_tokens=20)

    assert [c.section for c in chunks] == ["Power", "Status"]
    assert chunks[1].text.startswith("## Status")


def test_a_tiny_section_merges_into_the_next_and_the_bigger_one_names_the_chunk():
    text = f"### Keywords\n\nstatus; power\n\n## Introduction\n\n{paragraph(0, 8)}"
    chunks = chunk_pages(pages(text), max_tokens=200, min_tokens=20)

    assert len(chunks) == 1
    assert chunks[0].section == "Introduction"
    assert chunks[0].text.startswith("### Keywords")


def test_a_heading_is_never_separated_from_its_content():
    # A small section, then a heading followed by a paragraph that fills a chunk by itself.
    text = f"{paragraph(0, 2)}\n\n## Results\n\n{paragraph(2, 6)}"
    chunks = chunk_pages(pages(text), max_tokens=60, overlap_tokens=0, min_tokens=100)

    assert not any(c.text.split("\n\n")[-1].startswith("#") for c in chunks)  # none ends in one
    assert any(c.text.startswith("## Results") for c in chunks)  # it leads its section
