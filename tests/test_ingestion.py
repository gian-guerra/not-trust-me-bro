from pathlib import Path

import pymupdf
import pytest

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


FILLER = (
    "Groups differ in how they distribute resources and attention among their members. "
    "Observers notice these differences quickly and adjust their own behaviour to match. "
)


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    # Pages need a few paragraphs: the layout model treats a page holding one line as a header.
    body = [
        f"Status hierarchies form quickly in new groups.\n\n{FILLER}\n\n{FILLER}",
        f"Power changes how people attend to others.\n\n{FILLER}\n\n"
        "References\n\nSmith, J. (2020). A study of status. Journal of Groups, 1, 1-10.",
        "Jones, K. (2019). Another study. Journal of Power, 2, 11-20.\n\n"
        "Lee, M. (2018). A third study. Journal of Status, 3, 21-30.\n\n"
        "Park, S. (2017). A fourth study. Journal of Influence, 4, 31-40.",
    ]
    doc = pymupdf.open()
    for text in body:
        doc.new_page().insert_textbox(pymupdf.Rect(72, 150, 520, 600), text, fontsize=11)
    path = tmp_path / "sample.pdf"
    doc.save(path)
    return path


def test_load_pdf_returns_clean_pages_with_source_and_page_numbers(sample_pdf: Path):
    result = load_pdf(sample_pdf)

    assert [p.page_number for p in result] == [1, 2]
    assert all(p.source == "sample.pdf" for p in result)
    assert "Status hierarchies form quickly" in result[0].text
    assert "Power changes how people attend" in result[1].text
    assert "Smith" not in result[1].text
