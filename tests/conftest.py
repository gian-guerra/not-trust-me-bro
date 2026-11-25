from pathlib import Path

import pymupdf
import pytest

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
