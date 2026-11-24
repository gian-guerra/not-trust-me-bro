"""Ask a question end to end: retrieve the top-k chunks, then answer from them with citations.

A testing tool until PER-12 adds `ntmb ask`. Uses the settings in .env unless overridden.

Usage:
    uv run python scripts/ask.py "Why do people defer to high-status people?"
    uv run python scripts/ask.py "..." --think                 # local thinking mode
    uv run python scripts/ask.py "..." --model openai/gpt-6-luna
    uv run python scripts/ask.py "..." --mode bm25 --k 8 --show-sources
"""

import argparse
import time

from not_trust_me_bro.answers import answer
from not_trust_me_bro.config import Settings, get_settings
from not_trust_me_bro.embeddings import Embedder
from not_trust_me_bro.retrieval import Retriever
from not_trust_me_bro.store import ChunkStore, collection_name, connect


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("question")
    p.add_argument("--model", help="LiteLLM model, e.g. openai/gpt-6-luna (default: .env)")
    p.add_argument("--think", action="store_true", help="let local thinking models reason")
    p.add_argument("--mode", choices=["semantic", "bm25", "hybrid"])
    p.add_argument("--k", type=int)
    p.add_argument("--alpha", type=float)
    p.add_argument("--show-sources", action="store_true", help="print every chunk's full text")
    args = p.parse_args()

    base = get_settings()
    overrides = {
        "llm_model": args.model,
        "retrieval_mode": args.mode,
        "retrieval_k": args.k,
        "retrieval_alpha": args.alpha,
    }
    s = Settings(**{**base.model_dump(), **{k: v for k, v in overrides.items() if v is not None}})
    think = args.think or s.llm_think

    embedder = Embedder(s.embedding_model, s.embedding_query_prefix)
    with connect(s) as client:
        retriever = Retriever(ChunkStore(client, collection_name(s.embedding_model)), embedder)
        t0 = time.perf_counter()
        chunks = retriever.retrieve(
            args.question, s.retrieval_k, s.retrieval_mode, s.retrieval_alpha, s.retrieval_fusion
        )
        t1 = time.perf_counter()

    a = answer(args.question, chunks, s.llm_model, s.llm_temperature, think, s.llm_api_key())
    t2 = time.perf_counter()

    print(f"\nQ: {a.question}\n\n{a.text}\n")
    print(f"── sources ({s.retrieval_mode}, k={s.retrieval_k}) · ★ = cited")
    for i, c in enumerate(a.chunks, 1):
        star = "★" if c in a.cited else " "
        print(f" {star} [{i}] {c.score:6.3f}  {c.citation}  § {c.section[:50]}")
        if args.show_sources:
            print("      " + c.text.replace("\n", "\n      ") + "\n")

    flags = [
        name
        for name, on in [
            ("REFUSED", a.refused),
            ("UNCITED", a.uncited),
            (f"INVALID CITATIONS {a.invalid_citations}", a.invalid_citations),
        ]
        if on
    ]
    print(
        f"\n── {s.llm_model}{' (think)' if think else ''} · retrieval {t1 - t0:.1f}s · "
        f"answer {t2 - t1:.1f}s" + (f" · {', '.join(flags)}" if flags else "")
    )


if __name__ == "__main__":
    main()
