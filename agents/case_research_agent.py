"""
Case Research Agent
====================
Finds and summarizes relevant legal cases, judgments, and precedents
from the knowledge base or via structured web research prompts.
"""
from dataclasses import dataclass, field
from typing import List
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from core.llm import get_primary_llm
from rag.retriever import retrieve, format_context

DISCLAIMER = (
    "⚠️ Case summaries are for informational purposes. "
    "Always verify case details through official legal databases (e.g., Westlaw, LexisNexis, Google Scholar)."
)

_CASE_RESEARCH_PROMPT = PromptTemplate(
    input_variables=["context", "query"],
    template="""You are a legal researcher specializing in case law analysis. 
The user is looking for relevant cases and legal precedents related to their query.

RETRIEVED LEGAL CONTEXT (from knowledge base):
{context}

RESEARCH QUERY:
{query}

Provide a structured case research summary in EXACTLY this format:

RELEVANT_CASES:
- Case Name: [Full case name and citation if available]
  Court: [Court name and jurisdiction]
  Year: [Year decided]
  Key Holding: [The main legal ruling or principle established]
  Relevance: [Why this case is relevant to the query]

LEGAL_PRINCIPLES:
- [List key legal principles/doctrines relevant to this query]

JURISDICTIONAL_NOTES:
- [Note any important jurisdictional variations]

RESEARCH_SUMMARY:
[Plain-English summary of the case law landscape for this query, and how it applies]

Research:""",
)

_FALLBACK_PROMPT = PromptTemplate(
    input_variables=["query"],
    template="""You are a legal researcher. The user wants to find case law related to: {query}

Since no specific cases are in the knowledge base, provide:
1. The types of cases and legal principles typically relevant to this issue
2. Landmark cases in this area of law (from your training knowledge)
3. Guidance on how to search for authoritative case law
4. Key legal databases to consult (Westlaw, LexisNexis, Google Scholar, etc.)

Note clearly that these are general references and should be verified.

Response:""",
)


@dataclass
class CaseResult:
    query: str = ""
    cases: List[dict] = field(default_factory=list)
    legal_principles: List[str] = field(default_factory=list)
    jurisdictional_notes: List[str] = field(default_factory=list)
    summary: str = ""
    raw_research: str = ""
    used_knowledge_base: bool = False
    disclaimer: str = DISCLAIMER


def _parse_case_research(raw: str) -> tuple[list, list, list, str]:
    cases, principles, jnotes, summary = [], [], [], ""
    lines = raw.splitlines()
    section = None
    current_case: dict | None = None

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        if stripped == "RELEVANT_CASES:":
            section = "cases"
        elif stripped == "LEGAL_PRINCIPLES:":
            if current_case:
                cases.append(current_case)
                current_case = None
            section = "principles"
        elif stripped == "JURISDICTIONAL_NOTES:":
            section = "jurisdiction"
        elif stripped == "RESEARCH_SUMMARY:":
            section = "summary"

        elif section == "cases":
            if stripped.startswith("- Case Name:"):
                if current_case:
                    cases.append(current_case)
                current_case = {
                    "name": stripped[len("- Case Name:"):].strip(),
                    "court": "", "year": "", "holding": "", "relevance": "",
                }
            elif stripped.startswith("Court:") and current_case:
                current_case["court"] = stripped[len("Court:"):].strip()
            elif stripped.startswith("Year:") and current_case:
                current_case["year"] = stripped[len("Year:"):].strip()
            elif stripped.startswith("Key Holding:") and current_case:
                current_case["holding"] = stripped[len("Key Holding:"):].strip()
            elif stripped.startswith("Relevance:") and current_case:
                current_case["relevance"] = stripped[len("Relevance:"):].strip()

        elif section == "principles" and stripped.startswith("- "):
            principles.append(stripped[2:])

        elif section == "jurisdiction" and stripped.startswith("- "):
            jnotes.append(stripped[2:])

        elif section == "summary":
            summary += (" " + stripped) if summary else stripped

    if current_case:
        cases.append(current_case)

    return cases, principles, jnotes, summary


def research_cases(query: str, top_k: int = 6) -> CaseResult:
    """Research cases related to a legal query."""
    docs: List[Document] = retrieve(query, top_k=top_k)
    llm = get_primary_llm()

    if docs:
        context = format_context(docs)
        prompt = _CASE_RESEARCH_PROMPT.format(context=context, query=query)
        raw = llm.invoke(prompt)
        used_kb = True
    else:
        prompt = _FALLBACK_PROMPT.format(query=query)
        raw = llm.invoke(prompt)
        used_kb = False

    cases, principles, jnotes, summary = _parse_case_research(raw)
    return CaseResult(
        query=query,
        cases=cases,
        legal_principles=principles,
        jurisdictional_notes=jnotes,
        summary=summary,
        raw_research=raw,
        used_knowledge_base=used_kb,
    )
