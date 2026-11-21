"""Read chunks the way the retriever will see them. Statistics hide bugs; reading finds them.

Usage:
    uv run python scripts/preview_chunks.py data/paper.pdf                  # size summary
    uv run python scripts/preview_chunks.py data/paper.pdf --show 0 5 23    # print chunks
    uv run python scripts/preview_chunks.py data/paper.pdf --max-tokens 200 --overlap 40
"""

import argparse
from pathlib import Path

from not_trust_me_bro.chunking import chunk_pages, count_tokens
from not_trust_me_bro.ingestion import load_pdf


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("pdf", type=Path)
    p.add_argument("--show", nargs="*", type=int, default=[], help="chunk indexes to print")
    p.add_argument("--max-tokens", type=int, default=320)
    p.add_argument("--overlap", type=int, default=75)
    p.add_argument("--min-tokens", type=int, default=65)
    args = p.parse_args()

    chunks = chunk_pages(load_pdf(args.pdf), args.max_tokens, args.overlap, args.min_tokens)
    sizes = sorted(count_tokens(c.text) for c in chunks)
    print(
        f"{args.pdf.name}: {len(chunks)} chunks | tokens min {sizes[0]}, "
        f"median {sizes[len(sizes) // 2]}, max {sizes[-1]} | "
        f"spanning two pages: {sum(c.page_start != c.page_end for c in chunks)}"
    )
    for i in args.show:
        c = chunks[i]
        pages = f"p.{c.page_start}" + (f"-{c.page_end}" if c.page_end != c.page_start else "")
        print(f"\n{'=' * 20} #{i} | {pages} | § {c.section} | {count_tokens(c.text)} tokens")
        print(c.text)


if __name__ == "__main__":
    main()
