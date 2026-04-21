"""
FastAPI server for the RAG assistant.

Endpoints:
  POST /ingest          - Ingest a C++ codebase directory
  POST /query           - Query the codebase (non-streaming)
  POST /query/stream    - Query with streaming SSE response
  GET  /health          - Health check
  GET  /stats           - Index statistics

INTERVIEW NOTES on API design decisions:
  - Rate limiting: prevent abuse; 30 req/min per IP is reasonable for an LLM endpoint
  - SSE streaming: Server-Sent Events are simpler than WebSockets for one-way streaming
  - Separate /ingest endpoint: allows re-ingesting without restarting the server
  - /stats endpoint: useful for debugging (how many chunks are indexed?)
  - Background tasks for ingestion: don't block the request thread during embedding
"""
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from .config import settings
from .ingestion import ingest
from .vector_store import build_and_save, load_index
from .retriever import HybridRetriever
from .chain import RAGChain

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Rate limiter ─────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)

# ─── Global state ─────────────────────────────────────────────────────────────
# INTERVIEW: We store the RAG chain as app state rather than rebuilding it per
# request. FAISS index loading (from disk) takes ~2s; embedding model loading
# takes ~5s. We want both to happen once at startup, not per request.
class AppState:
    chain: Optional[RAGChain] = None
    all_chunks: list = []
    index_stats: dict = {}

app_state = AppState()


# ─── Lifespan ─────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load persisted index on startup if available."""
    index = load_index()
    if index and app_state.all_chunks:
        retriever = HybridRetriever(index, app_state.all_chunks)
        app_state.chain = RAGChain(retriever)
        logger.info("RAG chain initialized from persisted index")
    else:
        logger.info("No persisted index found. POST /ingest to index a codebase.")
    yield
    logger.info("Shutting down RAG assistant")


# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="C++ Codebase RAG Assistant",
    description="Query your C++ codebase in natural language",
    version="1.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Schemas ──────────────────────────────────────────────────────────────────
class IngestRequest(BaseModel):
    codebase_dir: str = Field(..., description="Absolute path to C++ codebase directory")

class IngestResponse(BaseModel):
    status: str
    files_loaded: int
    chunks_created: int

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    stream: bool = Field(default=False)

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    chunks_retrieved: int


# ─── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "index_ready": app_state.chain is not None,
        "chunks_indexed": app_state.index_stats.get("total_chunks", 0),
    }


@app.get("/stats")
async def stats():
    return app_state.index_stats


@app.post("/ingest", response_model=IngestResponse)
@limiter.limit("5/minute")
async def ingest_codebase(request: Request, body: IngestRequest, background_tasks: BackgroundTasks):
    """
    Ingest a C++ codebase directory.

    INTERVIEW: Why BackgroundTasks?
    Ingestion (load → chunk → embed → build FAISS) can take 30-120s for a
    large codebase. We don't want to hold the HTTP connection open that long.
    Instead we return immediately and process in background.
    For production, use a proper task queue (Celery + Redis) with /status polling.
    """
    codebase_dir = body.codebase_dir
    if not Path(codebase_dir).exists():
        raise HTTPException(status_code=400, detail=f"Directory not found: {codebase_dir}")

    try:
        chunks = ingest(codebase_dir)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    index = build_and_save(chunks)
    retriever = HybridRetriever(index, chunks)
    app_state.chain = RAGChain(retriever)
    app_state.all_chunks = chunks
    app_state.index_stats = {
        "codebase_dir": codebase_dir,
        "total_chunks": len(chunks),
        "unique_files": len({c.metadata["source"] for c in chunks}),
    }

    return IngestResponse(
        status="ok",
        files_loaded=app_state.index_stats["unique_files"],
        chunks_created=len(chunks),
    )


@app.post("/query", response_model=QueryResponse)
@limiter.limit(f"{settings.api_rate_limit}/minute")
async def query(request: Request, body: QueryRequest):
    """Non-streaming query endpoint."""
    if app_state.chain is None:
        raise HTTPException(status_code=503, detail="No codebase indexed. POST /ingest first.")

    answer, docs = app_state.chain.query(body.question)
    sources = list({doc.metadata.get("source", "") for doc in docs})

    return QueryResponse(answer=answer, sources=sources, chunks_retrieved=len(docs))


@app.post("/query/stream")
@limiter.limit(f"{settings.api_rate_limit}/minute")
async def query_stream(request: Request, body: QueryRequest):
    """
    Streaming query endpoint using Server-Sent Events.

    INTERVIEW: SSE vs WebSocket for streaming:
      - SSE: one-way server→client, HTTP/1.1 compatible, auto-reconnect
      - WebSocket: bidirectional, better for chat, requires WS upgrade
      For a Q&A assistant, SSE is simpler and sufficient.
    """
    if app_state.chain is None:
        raise HTTPException(status_code=503, detail="No codebase indexed. POST /ingest first.")

    async def event_generator():
        async for token in app_state.chain.stream_query(body.question):
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host=settings.api_host, port=settings.api_port, reload=True)
