"""Inspect PDFs before ingesting them: what does PyMuPDF actually extract?

Usage:
    uv run python scripts/inspect_pdfs.py <folder-or-pdf> [...]           # overview table
    uv run python scripts/inspect_pdfs.py <pdf> --pages 12 13 --raw       # dump pages
"""

import argparse
import statistics
from pathlib import Path

import pymupdf


def pdf_paths(targets: list[str]) -> list[Path]:
    paths: list[Path] = []
    for t in map(Path, targets):
        paths += sorted(t.rglob("*.pdf")) if t.is_dir() else [t]
    return paths


def overview(path: Path) -> dict:
    with pymupdf.open(path) as doc:
        chars = [len(page.get_text()) for page in doc]
        images = sum(len(page.get_images()) for page in doc)
        meta = doc.metadata or {}
        return {
            "file": path.name,
            "pages": doc.page_count,
            "title": (meta.get("title") or "").strip()[:60],
            "median_chars": int(statistics.median(chars)) if chars else 0,
            # Pages with almost no text are usually scans, figures or blank pages.
            "near_empty": sum(c < 100 for c in chars),
            "images": images,
        }


def dump(path: Path, pages: list[int], raw: bool) -> None:
    with pymupdf.open(path) as doc:
        for n in pages:  # 1-based, as a human reads page numbers
            text = doc[n - 1].get_text()
            print(f"\n{'=' * 30} {path.name} · page {n} · {len(text)} chars {'=' * 30}")
            print(repr(text) if raw else text)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("targets", nargs="+")
    p.add_argument("--pages", nargs="*", type=int, help="1-based page numbers to print")
    p.add_argument("--raw", action="store_true", help="show repr(): reveals \\n and odd chars")
    args = p.parse_args()

    paths = pdf_paths(args.targets)
    if args.pages:
        for path in paths:
            dump(path, args.pages, args.raw)
        return

    cols = ["file", "pages", "median_chars", "near_empty", "images", "title"]
    print(" | ".join(cols))
    for path in paths:
        row = overview(path)
        print(" | ".join(str(row[c]) for c in cols))


if __name__ == "__main__":
    main()
