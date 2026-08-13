"""
Legal text summarization using IBM watsonx.ai (map-reduce for long docs).
"""
from langchain.chains.summarize import load_summarize_chain
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from core.llm import get_primary_llm


_MAP_TEMPLATE = """You are a legal analyst. Summarize the following legal text excerpt.
Focus on: key obligations, rights, deadlines, and any unusual clauses.

Text:
{text}

Summary:"""

_COMBINE_TEMPLATE = """You are a senior legal analyst. Given these summaries of different sections
of a legal document, produce a single, coherent overall summary.
Highlight the most important points, obligations, and risks.

Summaries:
{text}

Final Summary:"""

MAP_PROMPT = PromptTemplate(template=_MAP_TEMPLATE, input_variables=["text"])
COMBINE_PROMPT = PromptTemplate(template=_COMBINE_TEMPLATE, input_variables=["text"])


def summarize_documents(docs: list[Document]) -> str:
    """Run map-reduce summarization over a list of document chunks."""
    llm = get_primary_llm()
    chain = load_summarize_chain(
        llm=llm,
        chain_type="map_reduce",
        map_prompt=MAP_PROMPT,
        combine_prompt=COMBINE_PROMPT,
        verbose=False,
    )
    result = chain.invoke({"input_documents": docs})
    return result.get("output_text", "").strip()


def summarize_text(text: str) -> str:
    """Convenience: summarize a raw text string directly."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from config.settings import get_settings
    settings = get_settings()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    docs = splitter.create_documents([text])
    if len(docs) == 1:
        # Short text — use a single prompt
        llm = get_primary_llm()
        prompt = MAP_PROMPT.format(text=text)
        return llm.invoke(prompt).strip()
    return summarize_documents(docs)
