"""Qdrant collection with named dense + sparse vectors and hybrid (RRF) search."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from qdrant_client import QdrantClient, models

from agentic_rag.indexing.embeddings import Embedder, SparseVec
from agentic_rag.ingestion.models import Chunk, Document

DENSE = "dense"
SPARSE = "sparse"


@dataclass
class SearchHit:
    chunk_id: str
    score: float
    text: str
    doc_id: str
    title: str
    source: str
    section_path: list[str]
    page_start: int | None
    page_end: int | None
    payload: dict[str, Any] = field(repr=False, default_factory=dict)

    @property
    def citation(self) -> str:
        pages = ""
        if self.page_start is not None:
            pages = (
                f" p.{self.page_start}"
                if self.page_start == self.page_end
                else f" pp.{self.page_start}-{self.page_end}"
            )
        section = f" § {self.section_path[-1]}" if self.section_path else ""
        return f"{self.title}{section}{pages}"


def _sparse(v: SparseVec) -> models.SparseVector:
    return models.SparseVector(indices=v.indices, values=v.values)


class QdrantStore:
    def __init__(self, client: QdrantClient, collection: str, embedder: Embedder) -> None:
        self.client = client
        self.collection = collection
        self.embedder = embedder

    # ------------------------------------------------------------ schema

    def ensure_collection(self) -> None:
        if self.client.collection_exists(self.collection):
            return
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config={
                DENSE: models.VectorParams(
                    size=self.embedder.dense_dim, distance=models.Distance.COSINE
                )
            },
            # BM25 needs IDF computed by Qdrant at query time across the collection.
            sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )
        for key in ("doc_id", "source"):
            self.client.create_payload_index(
                self.collection, field_name=key, field_schema=models.PayloadSchemaType.KEYWORD
            )

    # ------------------------------------------------------------ writes

    def upsert_document(
        self, doc: Document, chunks: Sequence[Chunk], embed_texts: Sequence[str]
    ) -> None:
        """Replace all chunks of a document atomically-enough: delete old, insert new."""
        self.delete_document(doc.doc_id)
        if not chunks:
            return
        dense, sparse = self.embedder.embed_documents(embed_texts)
        now = datetime.now(UTC).isoformat()
        points = [
            models.PointStruct(
                id=c.chunk_id,
                vector={DENSE: d, SPARSE: _sparse(s)},
                payload={
                    "doc_id": doc.doc_id,
                    "source": doc.source,
                    "title": doc.title,
                    "file_type": doc.file_type,
                    "content_hash": doc.content_hash,
                    "chunk_index": c.chunk_index,
                    "chunk_count": len(chunks),
                    "text": c.text,
                    "token_count": c.token_count,
                    "section_path": list(c.section_path),
                    "page_start": c.page_start,
                    "page_end": c.page_end,
                    "ingested_at": now,
                },
            )
            for c, d, s in zip(chunks, dense, sparse, strict=True)
        ]
        for i in range(0, len(points), 128):
            self.client.upsert(self.collection, points=points[i : i + 128], wait=True)

    def delete_document(self, doc_id: str) -> None:
        self.client.delete(
            self.collection,
            points_selector=models.FilterSelector(filter=_eq("doc_id", doc_id)),
            wait=True,
        )

    # ------------------------------------------------------------ reads

    def get_content_hash(self, doc_id: str) -> str | None:
        points, _ = self.client.scroll(
            self.collection, scroll_filter=_eq("doc_id", doc_id), limit=1, with_payload=True
        )
        if not points or not points[0].payload:
            return None
        value = points[0].payload.get("content_hash")
        return str(value) if value is not None else None

    def list_documents(self) -> dict[str, dict[str, Any]]:
        docs: dict[str, dict[str, Any]] = {}
        offset = None
        while True:
            points, offset = self.client.scroll(
                self.collection,
                limit=256,
                offset=offset,
                with_payload=["doc_id", "source", "title", "chunk_count", "ingested_at"],
            )
            for p in points:
                pl = p.payload or {}
                docs.setdefault(str(pl["doc_id"]), pl)
            if offset is None:
                return docs

    def count(self) -> int:
        return self.client.count(self.collection, exact=True).count

    def search(
        self,
        query: str,
        limit: int = 5,
        mode: str = "hybrid",
        prefetch_limit: int = 30,
        doc_ids: Sequence[str] | None = None,
    ) -> list[SearchHit]:
        """mode: 'hybrid' (dense + BM25 fused with RRF), 'dense' or 'sparse'."""
        dense, sparse = self.embedder.embed_query(query)
        flt = (
            models.Filter(
                must=[models.FieldCondition(key="doc_id", match=models.MatchAny(any=list(doc_ids)))]
            )
            if doc_ids
            else None
        )
        if mode == "dense":
            res = self.client.query_points(
                self.collection, query=dense, using=DENSE, limit=limit, query_filter=flt
            )
        elif mode == "sparse":
            res = self.client.query_points(
                self.collection, query=_sparse(sparse), using=SPARSE, limit=limit, query_filter=flt
            )
        elif mode == "hybrid":
            res = self.client.query_points(
                self.collection,
                prefetch=[
                    models.Prefetch(query=dense, using=DENSE, limit=prefetch_limit, filter=flt),
                    models.Prefetch(
                        query=_sparse(sparse), using=SPARSE, limit=prefetch_limit, filter=flt
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=limit,
            )
        else:
            raise ValueError(f"unknown search mode: {mode}")
        return [_to_hit(p) for p in res.points]


def _eq(key: str, value: str) -> models.Filter:
    return models.Filter(
        must=[models.FieldCondition(key=key, match=models.MatchValue(value=value))]
    )


def _to_hit(p: models.ScoredPoint) -> SearchHit:
    pl = p.payload or {}
    return SearchHit(
        chunk_id=str(p.id),
        score=p.score,
        text=str(pl.get("text", "")),
        doc_id=str(pl.get("doc_id", "")),
        title=str(pl.get("title", "")),
        source=str(pl.get("source", "")),
        section_path=list(pl.get("section_path") or []),
        page_start=pl.get("page_start"),
        page_end=pl.get("page_end"),
        payload=pl,
    )
