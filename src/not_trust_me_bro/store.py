"""Storage: chunks + their vectors in Weaviate (notes 06, 07).

One collection per embedding model. Vectors from different models live in different spaces:
comparing them gives meaningless scores, and two 384-dim models would not even raise an error.

Each chunk object keeps its text (indexed for BM25 keyword search), its citation metadata,
and its document's fingerprint, which makes re-ingestion idempotent (see pipeline.py).
"""

import re

import weaviate
from weaviate.classes.aggregate import GroupByAggregate
from weaviate.classes.config import Configure, DataType, Property, Tokenization, VectorDistances
from weaviate.classes.query import Filter
from weaviate.util import generate_uuid5

from not_trust_me_bro.chunking import Chunk
from not_trust_me_bro.config import Settings


def collection_name(embedding_model: str) -> str:
    """'BAAI/bge-small-en-v1.5' → 'Chunks_bge_small_en_v1_5'"""
    slug = re.sub(r"\W+", "_", embedding_model.split("/")[-1]).strip("_")
    return f"Chunks_{slug}"


def connect(settings: Settings) -> weaviate.WeaviateClient:
    return weaviate.connect_to_local(
        host=settings.weaviate_host,
        port=settings.weaviate_http_port,
        grpc_port=settings.weaviate_grpc_port,
    )


class ChunkStore:
    def __init__(self, client: weaviate.WeaviateClient, name: str) -> None:
        self.client = client
        self.name = name
        self._ensure_collection()
        self.collection = client.collections.get(name)

    def _ensure_collection(self) -> None:
        if self.client.collections.exists(self.name):
            return
        self.client.collections.create(
            self.name,
            # We send our own vectors; Weaviate builds the HNSW index over them (note 06).
            vector_config=Configure.Vectors.self_provided(
                vector_index_config=Configure.VectorIndex.hnsw(
                    distance_metric=VectorDistances.COSINE
                )
            ),
            properties=[
                # `word` tokenization feeds the BM25 index used by keyword and hybrid search.
                Property(name="text", data_type=DataType.TEXT, tokenization=Tokenization.WORD),
                # `field`: matched as a whole value, for exact filters like source == "x.pdf".
                Property(name="source", data_type=DataType.TEXT, tokenization=Tokenization.FIELD),
                Property(name="section", data_type=DataType.TEXT, skip_vectorization=True),
                Property(name="page_start", data_type=DataType.INT),
                Property(name="page_end", data_type=DataType.INT),
                Property(name="chunk_index", data_type=DataType.INT),
                Property(
                    name="fingerprint", data_type=DataType.TEXT, tokenization=Tokenization.FIELD
                ),
                Property(name="doc_chunks", data_type=DataType.INT),  # chunks in the document
            ],
        )

    def stored_fingerprint(self, source: str) -> str | None:
        """The fingerprint of a fully stored document, or None if missing or incomplete."""
        result = self.collection.query.fetch_objects(
            filters=Filter.by_property("source").equal(source), limit=1
        )
        if not result.objects:
            return None
        props = result.objects[0].properties
        complete = self.count(source) == props["doc_chunks"]  # e.g. a crash mid-insert
        return str(props["fingerprint"]) if complete else None

    def replace_document(
        self, chunks: list[Chunk], vectors: list[list[float]], fingerprint: str
    ) -> None:
        """Delete every chunk of the document, then insert the new ones."""
        if not chunks:
            return
        source = chunks[0].source
        self.delete_document(source)
        with self.collection.batch.fixed_size(batch_size=100) as batch:
            for chunk, vector in zip(chunks, vectors, strict=True):
                batch.add_object(
                    uuid=generate_uuid5(f"{source}:{chunk.chunk_index}"),
                    vector=vector,
                    properties={
                        "text": chunk.text,
                        "source": source,
                        "section": chunk.section or "",
                        "page_start": chunk.page_start,
                        "page_end": chunk.page_end,
                        "chunk_index": chunk.chunk_index,
                        "fingerprint": fingerprint,
                        "doc_chunks": len(chunks),
                    },
                )
        failed = self.collection.batch.failed_objects
        if failed:
            raise RuntimeError(f"{len(failed)} chunk(s) failed to insert: {failed[0].message}")

    def delete_document(self, source: str) -> None:
        self.collection.data.delete_many(where=Filter.by_property("source").equal(source))

    def count(self, source: str | None = None) -> int:
        filters = Filter.by_property("source").equal(source) if source else None
        return self.collection.aggregate.over_all(filters=filters, total_count=True).total_count

    def sources(self) -> dict[str, int]:
        """Every stored document and its number of chunks, by name."""
        result = self.collection.aggregate.over_all(
            group_by=GroupByAggregate(prop="source"), total_count=True
        )
        return dict(sorted((g.grouped_by.value, g.total_count) for g in result.groups))
