# not-trust-me-bro

> "I read it somewhere." "Someone told me." "Some website said so."
> Yeah... no.

**not-trust-me-bro** is a RAG (Retrieval-Augmented Generation) system for studying
and research that answers **only from sources you choose**: books, PDFs,
scientific papers, essays and notes. It doesn't trawl the open internet.

## Why

General-purpose LLMs answer from a mix of their training data and web searches.
You get fluent, generic answers, and you can't tell whether a claim comes from a
peer-reviewed paper or from a random forum post.

This project goes the other way:

- **You curate the library.** Only the material you add can be used.
- **Every answer cites its sources**: document, page and passage, so you can check it.
- **No source, no answer.** If your material doesn't cover the question, the system
  says so instead of making something up.

## What it will do

- **Ask questions** about your material and get grounded, cited answers
- **Study** a topic with generated summaries, flashcards and quizzes built
  from your sources
- **Research** across many documents: compare what different sources say and find
  where they agree or disagree

## How it works (high level)

```
Your sources ──► Ingest ──► Chunk ──► Embed ──► Vector store
                                                     │
Your question ──► Embed ──► Retrieve relevant chunks ┘
                                   │
                                   ▼
                     LLM answers using ONLY those chunks
                                   │
                                   ▼
                       Answer + citations
```

## Roadmap

1. **Ingestion**: load PDFs and split them into chunks
2. **Retrieval**: embeddings plus a vector store, returning the most relevant passages
3. **Grounded answers**: answers that cite document and page, and refuse when
   the answer isn't in the sources
4. **Evaluation**: a small test set to measure retrieval and answer quality
5. **Study tools**: summaries, flashcards and quizzes
6. **More formats**: EPUB, web articles, Markdown notes and more

## Status

🚧 Iteration 1: the core pipeline works end to end from the CLI. Evaluation is next.

## Usage

Requires [uv](https://docs.astral.sh/uv/), Docker, and either [Ollama](https://ollama.com) (`ollama pull qwen3:14b`) or an OpenAI key.

```bash
uv sync && cp .env.example .env
docker compose up -d                       # Weaviate

uv run ntmb ingest data/                   # a PDF or a folder; unchanged files are skipped
uv run ntmb sources                        # what's in your library
uv run ntmb ask "Why do people defer to high-status people?"
```

`ask` prints the answer with numbered citations and the passages behind them (★ = cited). If your sources don't cover the question, it says so instead of guessing. Useful options:

```bash
uv run ntmb ask "..." -s paper.pdf         # search one document (repeatable)
uv run ntmb ask "..." --think              # local model reasons first: slower, more careful
uv run ntmb ask "..." --model openai/gpt-6-luna   # needs NTMB_OPENAI_API_KEY in .env
uv run ntmb ask "..." --show-sources       # print the full text of every passage
```

Defaults live in `.env` (see `.env.example`); `uv run ntmb config` shows the current ones.

## Development

Requires [uv](https://docs.astral.sh/uv/) and Docker.

```bash
uv sync                  # install dependencies
cp .env.example .env     # local settings (git-ignored)
docker compose up -d     # start Weaviate on :8080 (REST) and :50051 (gRPC)
uv run ntmb --help       # the CLI
uv run pytest            # tests
uv run ruff check .      # lint
```

Put your source documents in `data/`. It's git-ignored, because most books and papers are copyrighted.

## License

[MIT](LICENSE)
