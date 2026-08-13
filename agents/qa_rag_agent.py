"""
Q&A RAG Agent
==============
Answers user legal questions using Retrieval-Augmented Generation.
Retrieves relevant chunks from the vector store, then synthesizes an answer.
"""
from dataclasses import dataclass, field
from typing import List
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from core.llm import get_primary_llm
from rag.retriever import retrieve, format_context

DISCLAIMER = (
    "⚠️ This answer is based on general legal information and does not constitute legal advice. "
    "Laws vary by jurisdiction and change over time. Consult a qualified attorney for your specific situation."
)

_QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a knowledgeable legal assistant. Use the provided legal context to answer the question accurately and in plain English.
If the context does not contain enough information to answer confidently, say so clearly — do not fabricate legal information.

LEGAL CONTEXT:
{context}

USER QUESTION:
{question}

Instructions:
- Provide a clear, structured answer
- Cite the source documents when relevant (e.g., "[Source 2]")
- Use plain language — avoid unnecessary jargon
- If jurisdiction-specific, note that laws may vary
- End with a brief note about any important caveats

Answer:""",
)


@dataclass
class QAResult:
    question: str = ""
    answer: str = ""
    sources: List[str] = field(default_factory=list)
    retrieved_chunks: int = 0
    disclaimer: str = DISCLAIMER


def answer_question(question: str, top_k: int = 6) -> QAResult:
    """Retrieve relevant context and generate a grounded answer."""
    docs: List[Document] = retrieve(question, top_k=top_k)
    context = format_context(docs)

    llm = get_primary_llm()
    prompt = _QA_PROMPT.format(context=context, question=question)
    answer = llm.invoke(prompt).strip()

    sources = list({
        doc.metadata.get("filename", doc.metadata.get("source", "Unknown"))
        for doc in docs
    })

    return QAResult(
        question=question,
        answer=answer,
        sources=sources,
        retrieved_chunks=len(docs),
    )
