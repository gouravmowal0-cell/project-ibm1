"""
Direct agent invocation routes — bypass the orchestrator for explicit calls.
"""
import dataclasses
from fastapi import APIRouter, HTTPException
from api.models import AgentRequest, AgentResponse

router = APIRouter(prefix="/agents", tags=["Agents"])


@router.post("/orchestrate", response_model=AgentResponse)
async def orchestrate(request: AgentRequest) -> AgentResponse:
    """
    Send any query to the orchestrator — it will classify intent and
    route to the correct agent automatically.
    """
    from agents.orchestrator import route
    result = route(
        user_input=request.query,
        session_id=request.session_id,
        document_text=request.document_text,
        framework=request.framework,
    )
    return AgentResponse(**result)


@router.post("/contract-review", response_model=AgentResponse)
async def contract_review(request: AgentRequest) -> AgentResponse:
    """Directly invoke the Contract Reviewer Agent."""
    if not request.document_text:
        raise HTTPException(status_code=400, detail="document_text is required for contract review.")
    from agents.contract_reviewer import review_contract
    result = review_contract(request.document_text)
    return AgentResponse(
        intent="CONTRACT",
        agent="contract_reviewer",
        data=dataclasses.asdict(result),
    )


@router.post("/compliance-check", response_model=AgentResponse)
async def compliance_check(request: AgentRequest) -> AgentResponse:
    """Directly invoke the Compliance Checker Agent."""
    if not request.document_text:
        raise HTTPException(status_code=400, detail="document_text is required for compliance check.")
    from agents.compliance_checker import check_compliance
    result = check_compliance(request.document_text, framework=request.framework)
    return AgentResponse(
        intent="COMPLIANCE",
        agent="compliance_checker",
        data=dataclasses.asdict(result),
    )


@router.post("/qa", response_model=AgentResponse)
async def qa(request: AgentRequest) -> AgentResponse:
    """Directly invoke the Q&A RAG Agent."""
    from agents.qa_rag_agent import answer_question
    result = answer_question(request.query)
    return AgentResponse(
        intent="QA",
        agent="qa_rag",
        data=dataclasses.asdict(result),
    )


@router.post("/case-research", response_model=AgentResponse)
async def case_research(request: AgentRequest) -> AgentResponse:
    """Directly invoke the Case Research Agent."""
    from agents.case_research_agent import research_cases
    result = research_cases(request.query)
    return AgentResponse(
        intent="CASE_RESEARCH",
        agent="case_research",
        data=dataclasses.asdict(result),
    )
