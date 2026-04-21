"""
FAISS vector store: build index from chunks, save/load, and search.

INTERVIEW NOTES on FAISS index types:
  - IndexFlatL2:  Exact nearest neighbour, brute force.
                  Perfect for < 100k vectors. O(n) per query.
  - IndexIVFFlat: Approximate NN using inverted file index.
                  Faster for > 100k vectors. Requires training step.
                  Trade-off: ~1-5% recall loss for 10-100x speed gain.

  For a typical C++ codebase (50k LOC → ~2k-5k chunks), IndexFlatL2
  is fast enough (<100ms) and gives perfect recall. We switch to IVF
  when the user sets FAISS_INDEX_TYPE=IVF in config.

INTERVIEW: Why FAISS over Pinecone/Weaviate?
  - Zero cost, runs locally, no network latency
  - For a single-tenant developer tool, managed vector DBs add complexity
    without meaningful benefit at this scale
"""
import logging
import os
from pathlib import Path

from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from .config import settings

logger = logging.getLogger(__name__)


def _get_embeddings() -> HuggingFaceEmbeddings:
    """
    Load HuggingFace sentence-transformer embeddings.

    INTERVIEW: all-MiniLM-L6-v2 is a strong default:
      - 384-dimensional vectors (compact, fast)
      - Trained on 1B+ sentence pairs
      - ~80MB model size
      - Free — no API key required
      - ~14ms per chunk on CPU

    For higher accuracy on code, consider:
      - microsoft/codebert-base (code-aware, larger)
      - nomic-ai/nomic-embed-text-v1 (longer context window)
    """
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": settings.embedding_device},
        encode_kwargs={"normalize_embeddings": True},  # cosine similarity
    )


def build_index(chunks: list[Document]) -> FAISS:
    """
    Embed all chunks and build a FAISS index from scratch.

    INTERVIEW: The embedding step is the bottleneck.
      - 5000 chunks × 14ms/chunk ≈ 70 seconds on CPU
      - One-time cost: we persist to disk so re-ingestion is not needed
      - In production: use GPU or batch parallelism
    """
    logger.info(f"Building FAISS index for {len(chunks)} chunks...")
    embeddings = _get_embeddings()
    index = FAISS.from_documents(chunks, embeddings)
    logger.info("FAISS index built successfully")
    return index


def save_index(index: FAISS) -> None:
    """Persist FAISS index to disk for reuse across restarts."""
    path = Path(settings.faiss_index_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    index.save_local(str(path))
    logger.info(f"FAISS index saved to {path}")


def load_index() -> FAISS | None:
    """Load persisted FAISS index. Returns None if not found."""
    path = Path(settings.faiss_index_path)
    if not path.exists():
        logger.info("No existing FAISS index found")
        return None
    embeddings = _get_embeddings()
    index = FAISS.load_local(
        str(path),
        embeddings,
        allow_dangerous_deserialization=True,  # required by LangChain for pickle
    )
    logger.info(f"FAISS index loaded from {path}")
    return index


def build_and_save(chunks: list[Document]) -> FAISS:
    """Convenience: build index, save to disk, and return."""
    index = build_index(chunks)
    save_index(index)
    return index
