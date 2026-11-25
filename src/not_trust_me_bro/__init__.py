"""not-trust-me-bro: a RAG system that answers only from your own sources."""

import logging
import warnings

import weaviate  # noqa: F401  (imported first on purpose, see below)

# Two harmless messages that dependencies print on every run, silenced by exact match:
# PyMuPDF's SWIG bindings trigger a Python 3.13 deprecation notice on import, and torch logs
# that it can't redirect subprocess output on macOS (we never start subprocesses).
# Weaviate calls simplefilter("default") on import, which re-enables every warning in the
# process, so our filter must be added after it.
warnings.filterwarnings("ignore", r"builtin type \w+ has no __module__", DeprecationWarning)
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
