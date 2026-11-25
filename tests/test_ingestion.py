from pathlib import Path

from not_trust_me_bro.ingestion import Page, drop_references, load_pdf, normalize


def pages(*texts: str) -> list[Page]:
    return [Page(source="doc.pdf", page_number=i, text=t) for i, t in enumerate(texts, 1)]


# --- normalize --------------------------------------------------------------------------


def test_normalize_folds_lookalike_characters():
    # non-breaking space, en space, ligature, backspace control char
    assert normalize("et\xa0al. the ﬁrst\x08 study") == "et al. the first study"


def test_normalize_collapses_whitespace_but_keeps_paragraphs():
    assert normalize("  one   two \n\n\n\n three  ") == "one two\n\nthree"


def test_normalize_keeps_meaningful_symbols():
    assert normalize("r = −0.07 ✓ “quoted”") == "r = −0.07 ✓ “quoted”"


# --- drop_references --------------------------------------------------------------------


def test_drop_references_keeps_text_before_heading_and_drops_the_list():
    result = drop_references(
        pages("Body.", "End of body.\n\n## **References**\n\nSmith, J. (2020).", "Jones (2019).")
    )
    assert [(p.page_number, p.text) for p in result] == [(1, "Body."), (2, "End of body.")]


def test_drop_references_keeps_tables_and_figures_after_the_list():
    result = drop_references(
        pages("Body.", "References\n\nSmith (2020).", "**Table 1**\n\n|a|b|", "Figure 2. A brain.")
    )
    assert [p.page_number for p in result] == [1, 3, 4]


def test_drop_references_ignores_the_word_inside_a_sentence():
    original = pages("See the references below for details.", "More body.")
    assert drop_references(original) == original


def test_drop_references_handles_a_heading_at_the_top_of_a_page():
    result = drop_references(pages("Body.", "# References\n\nSmith (2020)."))
    assert [p.page_number for p in result] == [1]


# --- load_pdf (end to end on a generated PDF) -------------------------------------------


def test_load_pdf_returns_clean_pages_with_source_and_page_numbers(sample_pdf: Path):
    result = load_pdf(sample_pdf)

    assert [p.page_number for p in result] == [1, 2]
    assert all(p.source == "sample.pdf" for p in result)
    assert "Status hierarchies form quickly" in result[0].text
    assert "Power changes how people attend" in result[1].text
    assert "Smith" not in result[1].text
