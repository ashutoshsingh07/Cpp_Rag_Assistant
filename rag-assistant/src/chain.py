"""
RAG chain: retrieval + prompt + LLM → answer.

INTERVIEW NOTES:

  The RAG pipeline:
    query → retrieve relevant chunks → build prompt with context → LLM → answer

  Why not just feed the whole codebase to the LLM?
    - GPT-4 context window: 128k tokens ≈ 100k chars
    - A typical C++ codebase: 500k-5M chars
    - Even with large context: full-codebase prompting is slow and expensive
    - RAG retrieves only the 3-5 most relevant chunks → focused, cheaper, faster

  Prompt engineering decisions:
    - System prompt instructs LLM to cite file sources
    - We include chunk metadata (filename, line count) in context
    - We ask the LLM to say "I don't know" if context is insufficient
      (prevents hallucination — critical for code analysis)
    - Streaming response: faster perceived latency for long answers
"""
import logging
from typing import AsyncGenerator

from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from .config import settings
from .retriever import HybridRetriever

logger = logging.getLogger(__name__)

# ─── Prompt ──────────────────────────────────────────────────────────────────
# INTERVIEW: The system prompt is the single most impactful thing you can tune.
# Key design decisions here:
#   1. "Only use the provided context" → prevents hallucination
#   2. "Cite the file name" → answers are traceable and verifiable
#   3. "If unsure, say so" → honesty over confident wrong answers

SYSTEM_PROMPT = """You are an expert C++ code analyst assistant.
You answer questions about a C++ codebase using ONLY the code context provided below.

Rules:
1. Base your answer strictly on the provided context. Do NOT hallucinate code that isn't shown.
2. For every code claim, cite the source file in parentheses, e.g. (config_loader.cpp).
3. If the context does not contain enough information, say: "The provided context doesn't cover this. Try asking about a specific file or function."
4. Format code snippets with ```cpp fences.
5. Be concise but complete. Explain the WHY, not just the WHAT.

Context (retrieved C++ code snippets):
{context}
"""

HUMAN_PROMPT = "{question}"

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", HUMAN_PROMPT),
])


# ─── LLM factory ─────────────────────────────────────────────────────────────
def _build_llm():
    """Instantiate LLM based on provider setting."""
    if settings.llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=settings.llm_model or "claude-3-5-haiku-20241022",
            anthropic_api_key=settings.anthropic_api_key,
            streaming=True,
        )
    else:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=settings.llm_model or "gpt-4o-mini",
            openai_api_key=settings.openai_api_key,
            streaming=True,
            temperature=0,  # Deterministic for code analysis
        )


# ─── Context builder ─────────────────────────────────────────────────────────
def _build_context(docs: list[Document]) -> str:
    """Format retrieved documents into LLM-readable context block."""
    parts = []
    for i, doc in enumerate(docs, 1):
        src = doc.metadata.get("source", "unknown")
        lines = doc.metadata.get("num_lines", "?")
        parts.append(
            f"[Chunk {i} — {src} ({lines} lines)]\n"
            f"```cpp\n{doc.page_content.strip()}\n```"
        )
    return "\n\n".join(parts)


# ─── RAG Chain ────────────────────────────────────────────────────────────────
class RAGChain:
    """
    Ties together retrieval and generation into a single callable.

    INTERVIEW: We keep retrieval and generation separate (not using
    LangChain's RetrievalQA chain) so we can:
      - Log retrieved chunks independently for debugging
      - Stream tokens directly without buffering
      - Swap retriever or LLM without changing the interface
    """

    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever
        self.llm = _build_llm()
        self._chain = PROMPT | self.llm | StrOutputParser()

    def query(self, question: str) -> tuple[str, list[Document]]:
        """
        Non-streaming query. Returns (answer, source_documents).
        Use for programmatic access where you need the full answer at once.
        """
        docs = self.retriever.retrieve(question)
        context = _build_context(docs)
        answer = self._chain.invoke({"question": question, "context": context})
        return answer, docs

    async def stream_query(
        self, question: str
    ) -> AsyncGenerator[str, None]:
        """
        Streaming query. Yields tokens as they arrive from the LLM.

        INTERVIEW: Streaming improves perceived latency significantly.
        For a 500-token answer, non-streaming waits ~3s before showing anything.
        Streaming shows the first token in ~300ms, making it feel much faster.
        """
        docs = self.retriever.retrieve(question)
        context = _build_context(docs)

        # Yield source files first so the UI can show them immediately
        sources = list({doc.metadata.get("source", "") for doc in docs})
        yield f"__SOURCES__:{','.join(sources)}\n"

        async for token in self._chain.astream(
            {"question": question, "context": context}
        ):
            yield token
