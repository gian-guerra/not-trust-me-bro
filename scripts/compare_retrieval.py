"""Run one question through every retrieval mode and compare the top-k side by side.

Usage:
    uv run python scripts/compare_retrieval.py "Why do people defer to high-status people?"
    uv run python scripts/compare_retrieval.py "..." --k 3 --alphas 0.25 0.5 0.75 --show-text
"""

import argparse

from not_trust_me_bro.config import get_settings
from not_trust_me_bro.embeddings import Embedder
from not_trust_me_bro.retrieval import Retriever
from not_trust_me_bro.store import ChunkStore, collection_name, connect


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("question")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--alphas", nargs="*", type=float, default=[0.5])
    p.add_argument("--show-text", action="store_true", help="print the start of each chunk")
    args = p.parse_args()

    s = get_settings()
    embedder = Embedder(s.embedding_model, s.embedding_query_prefix)
    runs = [("semantic", {}), ("bm25", {})]
    runs += [(f"hybrid α={a}", {"mode": "hybrid", "alpha": a}) for a in args.alphas]
    runs += [("hybrid RRF α=0.5", {"mode": "hybrid", "alpha": 0.5, "fusion": "ranked"})]

    with connect(s) as client:
        retriever = Retriever(ChunkStore(client, collection_name(s.embedding_model)), embedder)
        print(f"\nQ: {args.question}")
        for label, kwargs in runs:
            mode = kwargs.pop("mode", label)
            print(f"\n── {label}")
            for rank, c in enumerate(retriever.retrieve(args.question, args.k, mode, **kwargs), 1):
                pages = c.citation.split(", ")[-1]
                print(f"  {rank}. {c.score:7.3f}  {c.source[:18]:18} {pages:9} § {c.section[:42]}")
                if args.show_text:
                    print(f"       {c.text[:150].replace(chr(10), ' ')!r}")


if __name__ == "__main__":
    main()
