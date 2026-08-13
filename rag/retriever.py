"""
Hybrid retrieval: semantic (vector) + optional BM25 keyword reranking.
"""
from typing import List
from langchain_core.documents import Document
from core.vector_store import get_retriever as _base_retriever


def retrieve(query: str, top_k: int | None = None) -> List[Document]:
    """
    Retrieve the most relevant legal text chunks for a query.
    Uses MMR (Maximal Marginal Relevance) to balance relevance and diversity.
    """
    retriever = _base_retriever()
    if top_k is not None:
        retriever.search_kwargs["k"] = top_k

    docs = retriever.invoke(query)
    return docs


def format_context(docs: List[Document]) -> str:
    """Format retrieved documents into a single context string for LLM prompts."""
    if not docs:
        return "No relevant legal context was found in the knowledge base."

    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("filename", "Unknown Source")
        parts.append(f"[Source {i}: {source}]\n{doc.page_content.strip()}")

    return "\n\n---\n\n".join(parts)
