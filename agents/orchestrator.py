"""
Orchestrator Agent
===================
The master router. Classifies incoming requests and dispatches them
to the correct specialized agent using IBM Granite as the intent classifier.

Intent taxonomy:
  - CHAT          → Conversational legal assistant
  - CONTRACT      → Contract Reviewer Agent
  - COMPLIANCE    → Compliance Checker Agent
  - CASE_RESEARCH → Case Research Agent
  - QA            → Q&A RAG Agent (default fallback)
"""
from __future__ import annotations
import json
import logging
from typing import Any

from langchain_core.prompts import PromptTemplate
from core.llm import get_router_llm

logger = logging.getLogger(__name__)

INTENT_LABELS = {"CHAT", "CONTRACT", "COMPLIANCE", "CASE_RESEARCH", "QA"}

_ROUTER_PROMPT = PromptTemplate(
    input_variables=["user_input"],
    template="""Classify the following user request into exactly ONE of these categories:
- CHAT        : General legal question or conversational query
- CONTRACT    : Request to review, analyze, or check a contract or agreement
- COMPLIANCE  : Request to check regulatory compliance (GDPR, HIPAA, SOX, etc.)
- CASE_RESEARCH : Request to find legal cases, precedents, or judgments
- QA          : Specific legal question requiring a precise answer

User request: "{user_input}"

Respond with ONLY the category label. No explanation.
Category:""",
)


def classify_intent(user_input: str) -> str:
    """Use the fast router LLM to classify the user's intent."""
    llm = get_router_llm()
    prompt = _ROUTER_PROMPT.format(user_input=user_input[:500])
    raw = llm.invoke(prompt).strip().upper()

    # Extract just the label if the model added extra text
    for label in INTENT_LABELS:
        if label in raw:
            return label

    logger.warning("Could not classify intent from: %r — defaulting to QA", raw)
    return "QA"


def route(
    user_input: str,
    session_id: str = "default",
    document_text: str | None = None,
    framework: str = "GENERAL",
) -> dict[str, Any]:
    """
    Main orchestration entry point.

    Args:
        user_input:     The user's query or instruction.
        session_id:     Session ID for conversational continuity.
        document_text:  Pre-parsed document text (if user uploaded a file).
        framework:      Compliance framework (used when intent=COMPLIANCE).

    Returns:
        A dict with 'intent', 'agent', and the agent's result payload.
    """
    intent = classify_intent(user_input)
    logger.info("Routing intent=%s for input=%r", intent, user_input[:80])

    if intent == "CONTRACT" and document_text:
        from agents.contract_reviewer import review_contract
        result = review_contract(document_text)
        return {"intent": intent, "agent": "contract_reviewer", "data": _dataclass_to_dict(result)}

    if intent == "COMPLIANCE" and document_text:
        from agents.compliance_checker import check_compliance
        result = check_compliance(document_text, framework=framework)
        return {"intent": intent, "agent": "compliance_checker", "data": _dataclass_to_dict(result)}

    if intent == "CASE_RESEARCH":
        from agents.case_research_agent import research_cases
        result = research_cases(user_input)
        return {"intent": intent, "agent": "case_research", "data": _dataclass_to_dict(result)}

    if intent == "CHAT":
        from agents.conversational_assistant import chat
        result = chat(session_id=session_id, user_message=user_input)
        return {"intent": intent, "agent": "conversational_assistant", "data": result}

    # Default: Q&A RAG
    from agents.qa_rag_agent import answer_question
    result = answer_question(user_input)
    return {"intent": intent, "agent": "qa_rag", "data": _dataclass_to_dict(result)}


def _dataclass_to_dict(obj: Any) -> dict:
    """Recursively convert dataclasses to dicts for JSON serialization."""
    import dataclasses
    if dataclasses.is_dataclass(obj):
        return {
            k: _dataclass_to_dict(v)
            for k, v in dataclasses.asdict(obj).items()
        }
    if isinstance(obj, list):
        return [_dataclass_to_dict(i) for i in obj]
    return obj
