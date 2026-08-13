"""
ChromaDB vector store — persistent, collection-per-corpus.
"""
from functools import lru_cache
from langchain_chroma import Chroma
from core.embeddings import get_embeddings
from config.settings import get_settings


@lru_cache(maxsize=1)
def get_vector_store() -> Chroma:
    settings = get_settings()
    return Chroma(
        collection_name=settings.chroma_collection_name,
        embedding_function=get_embeddings(),
        persist_directory=settings.chroma_persist_dir,
    )


def get_retriever():
    settings = get_settings()
    return get_vector_store().as_retriever(
        search_type="mmr",  # Maximal Marginal Relevance — reduces redundancy
        search_kwargs={
            "k": settings.retrieval_top_k,
            "fetch_k": settings.retrieval_top_k * 3,
            "lambda_mult": 0.7,
        },
    )
