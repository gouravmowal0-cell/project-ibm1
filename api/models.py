"""
Pydantic request/response models shared across all API routes.
"""
from __future__ import annotations
from typing import Any, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------

class DisclaimerMixin(BaseModel):
    disclaimer: str = (
        "⚠️ This information is for general purposes only and does not constitute legal advice. "
        "Please consult a qualified attorney for advice specific to your situation."
    )


# ---------------------------------------------------------------------------
# Chat endpoint
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000, description="User's message")
    session_id: str = Field(default="default", description="Conversation session ID")


class ChatResponse(DisclaimerMixin):
    session_id: str
    response: str
    sources: List[str] = []
    turn: int = 0


# ---------------------------------------------------------------------------
# Document analysis endpoint
# ---------------------------------------------------------------------------

class DocumentAnalysisRequest(BaseModel):
    """Used when document text is sent as JSON (already parsed client-side)."""
    document_text: str = Field(..., min_length=10, description="Extracted document text")
    analysis_type: str = Field(
        default="CONTRACT",
        description="One of: CONTRACT, COMPLIANCE, SUMMARIZE",
    )
    framework: str = Field(default="GENERAL", description="Compliance framework (e.g. GDPR, HIPAA)")


class DocumentAnalysisResponse(DisclaimerMixin):
    analysis_type: str
    data: Any  # Typed result from the agent


# ---------------------------------------------------------------------------
# Agent endpoints
# ---------------------------------------------------------------------------

class AgentRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    session_id: str = "default"
    document_text: Optional[str] = None
    framework: str = "GENERAL"


class AgentResponse(DisclaimerMixin):
    intent: str
    agent: str
    data: Any


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"
    models: dict = {}
