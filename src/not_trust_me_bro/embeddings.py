"""Embeddings: text → vector, with sentence-transformers running locally (note 03).

Vectors are L2-normalized, so cosine similarity equals the dot product and every vector has
length 1. Documents and queries are embedded differently: BGE models expect an instruction
in front of short queries, never in front of documents.
"""

from sentence_transformers import SentenceTransformer


class TruncationError(ValueError):
    """A text is longer than the model's input limit. The model would silently ignore the
    overflow, so that part of the chunk could never be retrieved."""


class Embedder:
    def __init__(self, model_name: str, query_prefix: str = "") -> None:
        self.model_name = model_name
        self.query_prefix = query_prefix
        self._model = SentenceTransformer(model_name)
        self.dimensions: int = self._model.get_embedding_dimension()
        self.max_tokens: int = self._model.max_seq_length

    def count_tokens(self, text: str) -> int:
        """Exact token count with the model's own tokenizer, special tokens included."""
        return len(self._model.tokenizer(text, verbose=False)["input_ids"])

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        too_long = [
            (i, n) for i, t in enumerate(texts) if (n := self.count_tokens(t)) > self.max_tokens
        ]
        if too_long:
            raise TruncationError(
                f"{len(too_long)} text(s) exceed {self.model_name}'s {self.max_tokens}-token limit "
                f"and would be silently truncated (index, tokens): {too_long[:5]}"
            )
        vectors = self._model.encode(
            texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False
        )
        return vectors.tolist()

    def embed_query(self, query: str) -> list[float]:
        vector = self._model.encode(
            self.query_prefix + query, normalize_embeddings=True, show_progress_bar=False
        )
        return vector.tolist()
