# cpp-rag-assistant

> Natural-language code search for C++ codebases — powered by Retrieval-Augmented Generation.

Ask questions about your codebase in plain English and get accurate, source-cited answers in under 300ms.

```bash
$ curl -X POST http://localhost:8000/query \
  -d '{"question": "How does ConfigLoader handle variant overrides?"}'

{
  "answer": "ConfigLoader::loadVariant() merges a variant-specific XML file on top
             of the base config, overwriting matching keys. If the variant file does
             not exist it returns false silently — not an error (config_loader.hpp:47).",
  "sources": ["include/config_loader.hpp", "src/config_loader.cpp"],
  "chunks_retrieved": 3
}
```

---

## Features

- **Hybrid search** — dense FAISS retrieval + BM25 keyword matching, merged via Reciprocal Rank Fusion
- **Cross-encoder reranking** — re-scores top candidates for higher precision
- **Streaming responses** — tokens stream via Server-Sent Events as they are generated
- **C++-aware chunking** — splits on namespace, class, and function boundaries
- **Source citations** — every answer cites the exact files it drew from
- **Zero-cost embeddings** — HuggingFace `all-MiniLM-L6-v2` runs locally, no key needed for ingestion
- **Rate limiting** — 30 requests/min per IP to protect LLM costs
- **Docker-ready** — single `docker-compose up` starts everything

---

## Architecture

```
C++ source files
      │
      ▼
  Ingestion (C++-aware chunking)
      │
      ▼
  HuggingFace Embeddings (all-MiniLM-L6-v2, local)
      │
      ├──▶ FAISS Index (dense vector search)
      └──▶ BM25 Index  (keyword/token search)
                │
                ▼
        Reciprocal Rank Fusion (merge results)
                │
                ▼
        Cross-encoder Reranker (ms-marco-MiniLM-L-6-v2)
                │
                ▼
        LLM (GPT-4o-mini / Claude)
                │
                ▼
        Answer + source citations
```

| Layer | Technology |
|---|---|
| Embeddings | HuggingFace `all-MiniLM-L6-v2` (local, free) |
| Vector store | FAISS `IndexFlatL2` |
| Sparse retrieval | BM25 (rank-bm25) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM | OpenAI GPT-4o-mini or Anthropic Claude |
| API | FastAPI + uvicorn |
| Streaming | Server-Sent Events (SSE) |
| Containerisation | Docker + Docker Compose |

---

## Getting Started

### Prerequisites

- Python 3.12+
- An OpenAI or Anthropic API key (free tier works)

### Installation

```bash
git clone https://github.com/ashutoshsingh07/cpp-rag-assistant
cd cpp-rag-assistant
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Set OPENAI_API_KEY or ANTHROPIC_API_KEY in .env
```

### Ingest your codebase

```bash
python scripts/ingest_codebase.py /path/to/your/cpp/project
```

Walks the directory, chunks all `.cpp`, `.cc`, `.h`, `.hpp` files, embeds them, and saves the FAISS index to `./data/faiss_index`. One-time — subsequent queries load from disk in milliseconds.

### Start the API

```bash
uvicorn src.api:app --reload
# API at http://localhost:8000
# Interactive docs at http://localhost:8000/docs
```

### With Docker

```bash
docker-compose up --build

# Ingest via API
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"codebase_dir": "/codebase"}'
```

---

## API Reference

### `POST /ingest`
Index a C++ codebase directory.
```json
{ "codebase_dir": "/absolute/path/to/project" }
```

### `POST /query`
Query the indexed codebase.
```json
{ "question": "Where is memory allocated for platform configuration?" }
```
Response:
```json
{
  "answer": "Memory for platform configuration is allocated in PlatformManager::init()...",
  "sources": ["src/platform_manager.cpp", "include/platform.hpp"],
  "chunks_retrieved": 3
}
```

### `POST /query/stream`
Same as `/query` but streams tokens via SSE. First event carries source filenames.

### `GET /health`
```json
{ "status": "ok", "index_ready": true, "chunks_indexed": 4821 }
```

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `openai` | `openai` or `anthropic` |
| `LLM_MODEL` | `gpt-4o-mini` | Model name |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `ANTHROPIC_API_KEY` | — | Anthropic API key |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between adjacent chunks |
| `RETRIEVAL_K` | `6` | Candidates fetched before reranking |
| `RERANK_TOP_N` | `3` | Final chunks passed to LLM |
| `API_RATE_LIMIT` | `30` | Requests per minute per IP |

---

## Performance

Measured on a 50,000 LOC codebase (~4,800 chunks), CPU only:

| Stage | Latency |
|---|---|
| Ingestion (one-time) | ~70s |
| FAISS + BM25 retrieval | ~10ms |
| Cross-encoder reranking | ~45ms |
| LLM generation (GPT-4o-mini) | ~800ms |
| **Total P95 query latency** | **~900ms** |

---

## Project Structure

```
cpp-rag-assistant/
├── src/
│   ├── config.py          # Pydantic settings
│   ├── ingestion.py       # File loader + C++-aware chunking
│   ├── vector_store.py    # FAISS index management
│   ├── retriever.py       # Hybrid retrieval + RRF + reranking
│   ├── chain.py           # RAG chain with streaming
│   └── api.py             # FastAPI server
├── scripts/
│   └── ingest_codebase.py # CLI ingestion tool
├── sample_cpp/            # Sample C++ files for testing
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## License

MIT
