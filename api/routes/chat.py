"""
Chat route — multi-turn conversational legal assistant.
"""
import logging
from fastapi import APIRouter, HTTPException
from api.models import ChatRequest, ChatResponse
from config.settings import get_settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post("", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    """
    Send a message to the conversational legal assistant.
    Maintains session history across turns using `session_id`.
    """
    settings = get_settings()
    if not settings.ibm_watsonx_api_key or not settings.ibm_watsonx_project_id:
        raise HTTPException(
            status_code=503,
            detail=(
                "IBM watsonx.ai credentials are not configured. "
                "Please set IBM_WATSONX_API_KEY and IBM_WATSONX_PROJECT_ID "
                "in your Render environment variables."
            ),
        )
    try:
        from agents.conversational_assistant import chat
        result = chat(session_id=request.session_id, user_message=request.message)
        return ChatResponse(**result)
    except Exception as exc:
        logger.exception("Chat agent failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"LLM error: {exc}")


@router.delete("/{session_id}")
async def clear_chat_session(session_id: str):
    """Clear the conversation history for a given session."""
    from agents.conversational_assistant import clear_session
    clear_session(session_id)
    return {"message": f"Session '{session_id}' cleared."}
