"""
C++ codebase ingestion: load files → parse → chunk → return Documents.

INTERVIEW NOTES on chunking strategy:
  - Fixed-size chunking (naive): splits mid-function → loses context
  - Semantic chunking (what we use): splits on C++ structure boundaries
    (class, function, namespace) → chunks are semantically meaningful
  - We use RecursiveCharacterTextSplitter with C++-aware separators.
    The splitter tries each separator in order, falling back to the next
    only when the chunk is still too large.
"""
import os
import logging
from pathlib import Path
from typing import Generator

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from .config import settings

logger = logging.getLogger(__name__)

# C++-aware separators ordered from coarsest to finest.
# INTERVIEW: The order matters — we prefer to split at namespace/class
# boundaries first, preserving the largest meaningful unit.
CPP_SEPARATORS = [
    "\nnamespace ",      # namespace boundary
    "\nclass ",         # class definition
    "\nstruct ",        # struct definition
    "\n// ---",         # common section divider in C++ codebases
    "\nvoid ",          # function definitions
    "\nint ",
    "\nbool ",
    "\nauto ",
    "\n\n",             # paragraph / blank line
    "\n",               # line break
    " ",                # last resort: word boundary
]

CPP_EXTENSIONS = {".cpp", ".cc", ".cxx", ".c", ".h", ".hpp", ".hxx", ".hh"}


def _is_cpp_file(path: Path) -> bool:
    return path.suffix.lower() in CPP_EXTENSIONS


def _load_file(path: Path) -> str | None:
    """Read file content, skip binary or unreadable files."""
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        logger.warning(f"Could not read {path}: {e}")
        return None


def iter_cpp_files(codebase_dir: str) -> Generator[Path, None, None]:
    """Walk directory tree and yield all C++ source files."""
    root = Path(codebase_dir)
    if not root.exists():
        raise FileNotFoundError(f"Codebase directory not found: {codebase_dir}")

    for path in root.rglob("*"):
        if path.is_file() and _is_cpp_file(path):
            # Skip build artifacts, third-party, and hidden dirs
            parts = path.parts
            skip_dirs = {"build", "cmake-build-debug", "cmake-build-release",
                         ".git", "third_party", "vendor", "extern", "external"}
            if any(p in skip_dirs for p in parts):
                continue
            yield path


def load_codebase(codebase_dir: str) -> list[Document]:
    """
    Load all C++ files from a directory into LangChain Documents.

    Each Document carries metadata:
      - source: relative file path (shown to user in answers)
      - file_type: extension
      - num_lines: for relevance context

    INTERVIEW: Metadata is crucial — it lets us tell the user
    *which file* the answer came from, not just what the answer is.
    """
    documents: list[Document] = []
    root = Path(codebase_dir)

    for file_path in iter_cpp_files(codebase_dir):
        content = _load_file(file_path)
        if not content or not content.strip():
            continue

        rel_path = str(file_path.relative_to(root))
        doc = Document(
            page_content=content,
            metadata={
                "source": rel_path,
                "file_type": file_path.suffix,
                "num_lines": content.count("\n"),
                "file_size_bytes": file_path.stat().st_size,
            }
        )
        documents.append(doc)
        logger.debug(f"Loaded {rel_path} ({doc.metadata['num_lines']} lines)")

    logger.info(f"Loaded {len(documents)} C++ files from {codebase_dir}")
    return documents


def chunk_documents(documents: list[Document]) -> list[Document]:
    """
    Split loaded documents into smaller chunks for embedding.

    INTERVIEW: Why overlap?
      Overlap (chunk_overlap=200) ensures that context at chunk boundaries
      is not lost. If a function call spans the boundary of two chunks,
      the overlap means both chunks contain the call site, so retrieval
      works regardless of which chunk is fetched.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=CPP_SEPARATORS,
        length_function=len,
        is_separator_regex=False,
    )

    chunks = splitter.split_documents(documents)

    # Enrich metadata with chunk index for traceability
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["chunk_preview"] = chunk.page_content[:80].replace("\n", " ")

    logger.info(f"Split {len(documents)} files into {len(chunks)} chunks "
                f"(avg {sum(len(c.page_content) for c in chunks)//max(len(chunks),1)} chars/chunk)")
    return chunks


def ingest(codebase_dir: str) -> list[Document]:
    """Full ingestion pipeline: load → chunk → return."""
    docs = load_codebase(codebase_dir)
    if not docs:
        raise ValueError(f"No C++ files found in {codebase_dir}")
    return chunk_documents(docs)
