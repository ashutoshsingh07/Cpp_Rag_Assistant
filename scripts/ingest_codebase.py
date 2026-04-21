#!/usr/bin/env python3
"""
CLI script to ingest a C++ codebase and build the FAISS index.

Usage:
    python scripts/ingest_codebase.py /path/to/your/cpp/project
    python scripts/ingest_codebase.py /path/to/project --stats-only
"""
import sys
import argparse
import time
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion import ingest
from src.vector_store import build_and_save


def main():
    parser = argparse.ArgumentParser(description="Ingest a C++ codebase into FAISS")
    parser.add_argument("codebase_dir", help="Path to C++ codebase directory")
    parser.add_argument("--stats-only", action="store_true", help="Only show stats, don't build index")
    args = parser.parse_args()

    codebase_dir = args.codebase_dir
    if not Path(codebase_dir).exists():
        print(f"Error: Directory not found: {codebase_dir}")
        sys.exit(1)

    print(f"Ingesting codebase: {codebase_dir}")
    start = time.time()

    chunks = ingest(codebase_dir)

    files = len({c.metadata["source"] for c in chunks})
    total_chars = sum(len(c.page_content) for c in chunks)

    print(f"\n=== Ingestion Stats ===")
    print(f"Files loaded:      {files}")
    print(f"Chunks created:    {len(chunks)}")
    print(f"Total characters:  {total_chars:,}")
    print(f"Avg chunk size:    {total_chars // max(len(chunks), 1)} chars")
    print(f"Time elapsed:      {time.time() - start:.1f}s")

    if args.stats_only:
        return

    print(f"\nBuilding FAISS index...")
    embed_start = time.time()
    build_and_save(chunks)
    print(f"Index built in {time.time() - embed_start:.1f}s")
    print(f"\nDone. Start the API with: uvicorn src.api:app --reload")


if __name__ == "__main__":
    main()
