"""
Configuration management using pydantic-settings.
All values can be overridden via environment variables or .env file.
"""
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # LLM
    openai_api_key: str = Field(default="", description="OpenAI API key")
    anthropic_api_key: str = Field(default="", description="Anthropic API key")
    llm_provider: str = Field(default="openai", description="'openai' or 'anthropic'")
    llm_model: str = Field(default="gpt-4o-mini", description="Model name")

    # Embeddings (HuggingFace - free, no API key needed)
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="HuggingFace embedding model"
    )
    embedding_device: str = Field(default="cpu", description="'cpu' or 'cuda'")

    # FAISS vector store
    faiss_index_path: str = Field(default="./data/faiss_index", description="Path to save/load FAISS index")
    faiss_index_type: str = Field(default="Flat", description="'Flat' (exact) or 'IVF' (approx, faster for large codebases)")

    # Chunking
    # INTERVIEW NOTE: chunk_size=1000, overlap=200 is a common starting point.
    # For C++ code, we use larger chunks to keep function bodies intact.
    chunk_size: int = Field(default=1000, description="Characters per chunk")
    chunk_overlap: int = Field(default=200, description="Overlap between chunks to preserve context")

    # Retrieval
    retrieval_k: int = Field(default=6, description="Number of chunks to retrieve")
    rerank_top_n: int = Field(default=3, description="Chunks to pass to LLM after reranking")

    # API
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    api_rate_limit: int = Field(default=30, description="Requests per minute per IP")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Singleton — import this everywhere
settings = Settings()
