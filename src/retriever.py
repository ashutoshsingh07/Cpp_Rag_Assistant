"""
Retrieval + reranking pipeline.

INTERVIEW NOTES on retrieval strategy:

  1. Dense retrieval (vector similarity):
       Fast, semantic, handles paraphrasing well.
       Weakness: misses exact identifier names (e.g. "ConfigLoader::load").

  2. Sparse retrieval (BM25 keyword search):
       Great for exact token matches (variable names, function signatures).
       Weakness: no semantic understanding.

  3. Hybrid retrieval (what we use):
       Combine dense + sparse scores using Reciprocal Rank Fusion (RRF).
       Gets the best of both: semantic understanding + exact matches.
       RRF formula: score(d) = Σ 1/(k + rank_i(d)), k=60 is standard.

  4. Reranking (cross-encoder):
       After hybrid retrieval fetches top-K candidates, a cross-encoder
       reranker scores each (query, chunk) pair jointly — more accurate
       than embedding cosine similarity but too slow for full index.
       We use a lightweight cross-encoder (ms-marco-MiniLM-L-6-v2).

INTERVIEW: Why not just use dense retrieval?
  In C++ codebases, queries like "find all uses of ConfigLoader::load"
  require exact token matching. Pure dense retrieval would return
  semantically similar but not necessarily exact matches. Hybrid + rerank
  significantly improves precision for code search.
"""
import logging
from dataclasses import dataclass
from typing import Optional

from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever

from .config import settings

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    document: Document
    score: float
    rank: int


def _reciprocal_rank_fusion(
    dense_results: list[Document],
    sparse_results: list[Document],
    k: int = 60,
) -> list[Document]:
    """
    Merge dense + sparse ranked lists using Reciprocal Rank Fusion.

    INTERVIEW: RRF(k=60) is the de-facto standard for hybrid search.
    k=60 was empirically found to give robust results across many benchmarks.
    Higher k → less penalty for lower ranks → more conservative merging.
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for rank, doc in enumerate(dense_results):
        key = doc.metadata.get("source", "") + str(doc.metadata.get("chunk_id", ""))
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        doc_map[key] = doc

    for rank, doc in enumerate(sparse_results):
        key = doc.metadata.get("source", "") + str(doc.metadata.get("chunk_id", ""))
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        doc_map[key] = doc

    sorted_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [doc_map[k] for k in sorted_keys]


class HybridRetriever:
    """
    Combines FAISS dense retrieval with BM25 sparse retrieval,
    then optionally reranks results with a cross-encoder.
    """

    def __init__(
        self,
        vector_store: FAISS,
        all_chunks: list[Document],
        use_reranker: bool = True,
    ):
        self.vector_store = vector_store
        self.use_reranker = use_reranker
        self._reranker = None

        # Build BM25 index over all chunks
        # INTERVIEW: BM25Retriever builds an in-memory inverted index.
        # For 5000 chunks it's instantaneous; for 100k+ chunks consider
        # Elasticsearch or OpenSearch with BM25 built in.
        self.bm25 = BM25Retriever.from_documents(all_chunks)
        self.bm25.k = settings.retrieval_k

        if use_reranker:
            self._load_reranker()

    def _load_reranker(self) -> None:
        """
        Load cross-encoder reranker (lazy — only if use_reranker=True).

        INTERVIEW: Cross-encoders process query+document jointly through
        a transformer, giving much better relevance scores than dot-product
        similarity. But they're O(k) transformer forward passes, so we
        only apply them to the top-K retrieved candidates, not the full index.
        """
        try:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(
                "cross-encoder/ms-marco-MiniLM-L-6-v2",
                max_length=512,
            )
            logger.info("Cross-encoder reranker loaded")
        except ImportError:
            logger.warning("sentence-transformers not installed; skipping reranker")
            self.use_reranker = False

    def _rerank(self, query: str, docs: list[Document]) -> list[Document]:
        """Score (query, doc) pairs and return sorted by relevance."""
        if not self._reranker or not docs:
            return docs

        pairs = [(query, doc.page_content) for doc in docs]
        scores = self._reranker.predict(pairs)

        ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in ranked[: settings.rerank_top_n]]

    def retrieve(self, query: str) -> list[Document]:
        """
        Full retrieval pipeline:
          1. Dense retrieval (FAISS cosine similarity)
          2. Sparse retrieval (BM25 keyword match)
          3. Hybrid merge via RRF
          4. Cross-encoder reranking
        """
        k = settings.retrieval_k

        # Step 1: Dense
        dense_docs = self.vector_store.similarity_search(query, k=k)

        # Step 2: Sparse
        sparse_docs = self.bm25.get_relevant_documents(query)

        # Step 3: Merge
        merged = _reciprocal_rank_fusion(dense_docs, sparse_docs)

        # Step 4: Rerank
        if self.use_reranker:
            return self._rerank(query, merged[: k * 2])

        return merged[: settings.rerank_top_n]
