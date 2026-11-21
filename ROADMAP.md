# Roadmap

## Iteration 1: core pipeline

**Goal:** ask a question about your own English PDFs from the CLI and get an answer that
cites document and page, or a clear "not in your sources". Then measure how good it is.

**Approach:** build the thinnest working version of every stage first (tasks 1–7), then
measure it (8–9), then improve one stage at a time against the measurements (10).
Core pieces such as chunking, retrieval metrics and the prompt are written by hand, with no
LangChain or LlamaIndex, so each concept stays visible.

### Stack

| Layer | Choice |
|---|---|
| Language and tooling | Python 3.13, uv, ruff, pytest |
| PDF parsing | PyMuPDF |
| Chunking | own code |
| Embeddings | sentence-transformers, English-only model, run locally |
| Vector DB | Weaviate (Docker): dense vectors, BM25 and hybrid search |
| LLM | LiteLLM; Ollama locally (`qwen3:14b`, `gemma4:12b-mlx`), OpenAI as an option |
| Evaluation | own recall@k / MRR script; RAGAS later |
| CLI | Typer |
| Config | pydantic-settings + `.env` |

### Tasks

| # | Task | Stage | Done when |
|---|---|---|---|
| 0 | Project setup | — | `uv run ntmb --help` works, Weaviate starts with `docker compose up` |
| 1 | Pick a starter corpus and inspect it | data | 2–3 PDFs chosen, extracted text reviewed, problems noted |
| 2 | PDF ingestion | ingestion | PDF → list of pages with text, file name and page number |
| 3 | Chunking | ingestion | pages → chunks with size/overlap settings, each keeps its source and page |
| 4 | Embeddings + storage | storage | chunks embedded and stored in Weaviate; re-ingesting a file doesn't duplicate it |
| 5 | Retrieval | retrieval | question → top-k chunks with scores (semantic first) |
| 6 | Grounded answers | answers | answer cites `[file, p. X]`; refuses when the chunks don't cover the question |
| 7 | CLI | interface | `ntmb ingest <pdf>` and `ntmb ask "<question>"` work end to end |
| 8 | Evaluation set | evaluation | 15–20 questions, each with the source page(s) that answer it |
| 9 | Retrieval metrics | evaluation | script reports recall@k and MRR for the current settings |
| 10 | Experiments | all | compare chunk sizes, two embedding models, semantic vs hybrid (alpha), k, and local vs OpenAI LLM; results logged in `docs/experiments.md` |

## Later iterations (not planned in detail yet)

- Better retrieval: reranking with a cross-encoder, query rewriting
- Answer evaluation with RAGAS (faithfulness, relevancy)
- Observability: tracing each request
- Study tools: summaries, flashcards, quizzes
- More formats: EPUB, web articles, Markdown notes
- API (FastAPI), then a front-end and an agent on top of the same core library

## Note on data

Your source files (books, papers) live in `data/`, which is git-ignored. Most of them are
copyrighted, and this repo is public.
