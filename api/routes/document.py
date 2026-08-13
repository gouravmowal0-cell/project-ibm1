"""
Document upload & analysis route.
Accepts multipart file uploads (PDF, DOCX, TXT) or raw JSON text.
"""
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from api.models import DocumentAnalysisRequest, DocumentAnalysisResponse
import dataclasses

router = APIRouter(prefix="/document", tags=["Document"])


@router.post("/upload", response_model=DocumentAnalysisResponse)
async def upload_and_analyze(
    file: UploadFile = File(...),
    analysis_type: str = Form(default="CONTRACT"),
    framework: str = Form(default="GENERAL"),
) -> DocumentAnalysisResponse:
    """
    Upload a document (PDF/DOCX/TXT) and run the requested analysis.
    - analysis_type=CONTRACT  → Contract Reviewer Agent
    - analysis_type=COMPLIANCE → Compliance Checker Agent
    - analysis_type=SUMMARIZE  → Legal Summarizer
    """
    from parsers.document_parser import parse_bytes
    content = await file.read()
    try:
        text = parse_bytes(file.filename or "upload.txt", content)
    except ValueError as exc:
        raise HTTPException(status_code=415, detail=str(exc))

    return _run_analysis(text, analysis_type.upper(), framework.upper())


@router.post("/analyze", response_model=DocumentAnalysisResponse)
async def analyze_text(request: DocumentAnalysisRequest) -> DocumentAnalysisResponse:
    """Analyze a document supplied as plain text in the request body."""
    return _run_analysis(
        request.document_text,
        request.analysis_type.upper(),
        request.framework.upper(),
    )


def _run_analysis(text: str, analysis_type: str, framework: str) -> DocumentAnalysisResponse:
    if analysis_type == "CONTRACT":
        from agents.contract_reviewer import review_contract
        result = review_contract(text)
        data = dataclasses.asdict(result)

    elif analysis_type == "COMPLIANCE":
        from agents.compliance_checker import check_compliance
        result = check_compliance(text, framework=framework)
        data = dataclasses.asdict(result)

    elif analysis_type == "SUMMARIZE":
        from rag.summarizer import summarize_text
        summary = summarize_text(text)
        data = {"summary": summary}

    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown analysis_type '{analysis_type}'. Use CONTRACT, COMPLIANCE, or SUMMARIZE.",
        )

    return DocumentAnalysisResponse(analysis_type=analysis_type, data=data)
