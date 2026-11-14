"""Dense + sparse (BM25) embeddings via fastembed (ONNX, CPU, open-weight)."""

from collections.abc import Sequence

from agentic_rag.indexing.embedder import SparseVec


class FastEmbedEmbedder:
    """bge-small (dense, 384-d) + Qdrant/bm25 (sparse). Models load lazily on first use."""

    def __init__(self, dense_model: str, sparse_model: str, batch_size: int = 32) -> None:
        from fastembed import SparseTextEmbedding, TextEmbedding

        self._dense = TextEmbedding(dense_model)
        self._sparse = SparseTextEmbedding(sparse_model)
        self._batch_size = batch_size
        self.dense_dim = len(next(iter(self._dense.query_embed("dimension probe"))))

    def embed_documents(self, texts: Sequence[str]) -> tuple[list[list[float]], list[SparseVec]]:
        dense = [v.tolist() for v in self._dense.embed(list(texts), batch_size=self._batch_size)]
        sparse = [
            SparseVec(s.indices.tolist(), s.values.tolist())
            for s in self._sparse.embed(list(texts), batch_size=self._batch_size)
        ]
        return dense, sparse

    def embed_query(self, text: str) -> tuple[list[float], SparseVec]:
        # query_embed applies the model's query prefix / query-side BM25 weighting.
        dense = next(iter(self._dense.query_embed(text))).tolist()
        s = next(iter(self._sparse.query_embed(text)))
        return dense, SparseVec(s.indices.tolist(), s.values.tolist())
