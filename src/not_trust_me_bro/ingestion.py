"""Ingestion: turn a PDF into clean pages, each tagged with its source and page number.

Pipeline:  PDF ──extract──► Markdown per page ──normalize──► ──drop references──► pages

Decisions come from inspecting the starter corpus (PER-6):
- pymupdf4llm instead of plain get_text(): it rebuilds tables as Markdown, joins wrapped lines
  into paragraphs, keeps headings as `#`, and (header/footer=False) drops running headers,
  footers and page numbers.
- Reference lists were ~45% of the pages and would flood retrieval, so they are dropped,
  except for table/figure pages that some papers place after the references.

Known limitations of pymupdf4llm (checked on the starter corpus: 95.5–99.7% of words kept,
and every loss was page furniture):
- It joins words hyphenated across a line break by deleting the hyphen ("meta-analysis" →
  "metaanalysis"). Rare, harmless for embeddings, but BM25 won't match the hyphenated spelling.
- A page holding a single line of text is classified as a header and dropped.
"""

import re
import unicodedata
from dataclasses import dataclass, replace
from pathlib import Path

import pymupdf4llm


@dataclass(frozen=True)
class Page:
    source: str  # file name, used in citations
    page_number: int  # 1-based, as printed in a citation
    text: str  # Markdown


def extract_pages(path: Path) -> list[Page]:
    """Extract every page of a PDF as Markdown."""
    chunks = pymupdf4llm.to_markdown(
        str(path),
        page_chunks=True,  # one result per page, so every piece of text keeps its page number
        header=False,
        footer=False,
        use_ocr=False,  # every PDF so far has a text layer; OCR is slow and not needed
        show_progress=False,
    )
    return [
        Page(source=path.name, page_number=c["metadata"]["page_number"], text=c["text"])
        for c in chunks
    ]


_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")  # keeps \t and \n


def normalize(text: str) -> str:
    """Make text consistent without changing its meaning.

    NFKC folds look-alike characters into one form: non-breaking and en spaces become plain
    spaces, ligatures like "ﬁ" become "fi". Different bytes for the same word would otherwise
    count as different words for BM25.
    """
    text = unicodedata.normalize("NFKC", text)
    text = _CONTROL_CHARS.sub("", text)
    text = re.sub(r"[ \t]+", " ", text)  # collapse runs of spaces
    text = re.sub(r" *\n *", "\n", text)  # no spaces around line breaks
    text = re.sub(r"\n{3,}", "\n\n", text)  # at most one blank line between paragraphs
    return text.strip()


# A line that is only the word, optionally wrapped in Markdown: "# **References**", "References"
_REFERENCES_HEADING = re.compile(
    r"^[#*_\s]*(references|bibliography|works cited)[*_\s]*$", re.IGNORECASE | re.MULTILINE
)
_TABLE_OR_FIGURE = re.compile(r"^[#*_\s]*(table|fig\.?|figure)\s", re.IGNORECASE)


def drop_references(pages: list[Page]) -> list[Page]:
    """Remove the reference list, keeping body text before it and table/figure pages after it."""
    found = next(
        ((i, m) for i, p in enumerate(pages) if (m := _REFERENCES_HEADING.search(p.text))),
        None,
    )
    if found is None:
        return pages  # no reference list

    i, heading = found
    page = pages[i]
    before = page.text[: heading.start()].strip()
    kept = pages[:i] + ([replace(page, text=before)] if before else [])
    kept += [p for p in pages[i + 1 :] if _TABLE_OR_FIGURE.match(p.text)]
    return kept


def load_pdf(path: Path) -> list[Page]:
    """Extract, normalize and clean a PDF. Empty pages are dropped."""
    pages = [replace(p, text=normalize(p.text)) for p in extract_pages(path)]
    return [p for p in drop_references(pages) if p.text]
